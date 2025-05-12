#!/usr/bin/env python

"""
CARLA Data Collection Script

This script collects RGB, depth, and semantic segmentation data from left, center, and right cameras
while the ego vehicle is driven by CARLA's autopilot. It also spawns other vehicles and pedestrians
to create a realistic environment.
"""

import glob
import os
import re
import sys
import time
import random
import argparse
import subprocess
import logging
import tqdm
import textwrap
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent

import numpy as np

def get_actor_blueprints(world, filter, generation):
    """Returns a list of blueprints based on filter and generation specifications"""
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, assume it's the one needed
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

class CarlaSensorDataCollector:
    def __init__(self, args):
        """Initialize the data collector with specified arguments"""
        self.client = None
        self.world = None
        self.map_spawn_points = None
        self.traffic_manager = None
        self.ego_vehicle = None
        self.autonomous_agent = None
        self.sensors = []
        self.actor_list = []
        self.pedestrian_controllers = []
        self.vehicles_list = []
        self.walkers_list = []

        # Assign args to class attributes
        self.args = args
        self.output_dir = args.output_dir
        self.map_name = args.map
        self.weather = args.weather
        self.number_of_vehicles = args.number_of_vehicles
        self.number_of_walkers = args.number_of_walkers
        self.sync = args.sync
        self.no_render = args.no_render
        self.spawn_point_idx = args.spawn_point

        # Create output directories
        self._setup_output_directories()

        # Docker setup
        self.docker_image = None
        self.docker = None

        # Keep track of frame count
        self.frame_count = 0
        self.sensor_data_received = {}  # Track which sensors have reported
        self.sensor_data = {}  # Store sensor data before writing to disk
        self.num_sensors = 9  # Total number of sensors we expect (3 RGB, 3 depth, 3 semantic)
        self.frame_ready = False  # Flag to indicate if a frame is complete

    def _setup_output_directories(self):
        """Create output directories for different sensor types if they don't exist"""
        # Main output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Subdirectories for different sensor types and perspectives
        perspectives = ['left', 'central', 'right']
        sensor_types = ['RGB', 'DEPTH', 'SS']
        additional_sensor_types = ['CAN']

        for sensor_type in sensor_types:
            for perspective in perspectives:
                dir_path = os.path.join(self.output_dir, sensor_type, perspective)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

    def start(self):
        """Start the CARLA client and set up the simulation"""
        try:
            # Set up Docker/server
            if self.docker_image is not None:
                self._setup_docker()

            # Connect to the client and get world
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(10.0)

            # Load desired map if specified and different from current
            # available_maps = self.client.get_available_maps()
            # print("Available Maps:", available_maps)
            current_map = self.client.get_world().get_map().name
            # desired_map = f"/Game/Carla/Maps/{self.map_name}" if not self.map_name.startswith('/') else self.map_name
            # desired_map = self.map_name

            if not current_map.endswith(self.map_name):
                print(f"Loading map {self.map_name}...")
                self.world = self.client.load_world(self.map_name)
            else:
                print(f"Map {self.map_name} is already loaded, using current map {current_map}")
                self.world = self.client.get_world()
            # Reload to fix a bug in CARLA 0.9.15 w/pedestrians not adhering to surface
            # else:
            #     self.world = self.client.get_world()
            #     if desired_map != current_map:
            #         print(f"Warning: Map {desired_map} not found or already loaded, using current map {current_map}")

            # Set up synchronous mode if requested
            settings = self.world.get_settings()
            if self.sync:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.args.fps

            if self.no_render:
                settings.no_rendering_mode = True

            self.world.apply_settings(settings)

            # Set the weather if specified
            if self.weather is not None:
                if not hasattr(carla.WeatherParameters, self.weather):
                    print('ERROR: weather preset %r not found.' % self.weather)
                else:
                    print('set weather preset %r.' % self.weather)
                    self.world.set_weather(getattr(carla.WeatherParameters, self.weather))

            # Set up traffic manager
            self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

            if self.sync:
                self.traffic_manager.set_synchronous_mode(True)

            # Configure traffic manager behavior
            self.traffic_manager.set_hybrid_physics_mode(True)

            # Set random seed, if provided
            if self.args.seed is not None:
                self.traffic_manager.set_random_device_seed(self.args.seed)
                # Crucial before spawning actors: seed everything!
                random.seed(self.args.seed)

            # Spawn ego vehicle, pedestrians, and other vehicles
            self._spawn_ego_vehicle()
            self._setup_sensors()
            self._spawn_other_vehicles()
            self._spawn_pedestrians()

            # Start the simulation loop
            self._simulation_loop()

        except KeyboardInterrupt:
            print('\nCancelled by user.')
        finally:
            self.cleanup()

    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at a specified spawn point and set up the agent"""
        # Get the blueprint for the ego vehicle (hardcoded: Lincoln MKZ 2017)
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2017')
        vehicle_bp.set_attribute('role_name', 'hero')

        # Find a valid spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise ValueError("No spawn points available in the map")

        # Use specific spawn point if provided, otherwise choose random
        if self.spawn_point_idx >= 0 and self.spawn_point_idx < len(spawn_points):
            spawn_point = spawn_points[self.spawn_point_idx]
        else:
            spawn_point = random.choice(spawn_points)

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.ego_vehicle)
        print(f"Spawned ego vehicle at spawn point index {self.spawn_point_idx}, coordinates "
              f"(x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}, z={spawn_point.location.z:.2f})")

        # Default: set autopilot
        if self.args.agent == "Autopilot":
            # Standard autopilot from the Traffic Manager (already handled in _spawn_ego_vehicle)
            print(f"Using built-in autopilot")
            self.ego_vehicle.set_autopilot(True, self.args.tm_port)
            return

        # Disable built-in autopilot if using a custom agent
        self.ego_vehicle.set_autopilot(False)

        if self.args.agent == "Basic":
            print(f"Setting up Basic agent")
            self.autonomous_agent = BasicAgent(self.ego_vehicle, target_speed=self.args.max_speed)
            self.autonomous_agent.follow_speed_limits(True)
        elif self.args.agent == "Constant":
            print(f"Setting up Constant Velocity agent")
            self.autonomous_agent = ConstantVelocityAgent(self.ego_vehicle, target_speed=self.args.max_speed)
            ground_loc = self.world.ground_projection(self.ego_vehicle.get_location(), 5)
            if ground_loc:
                self.ego_vehicle.set_location(ground_loc.location + carla.Location(z=0.01))
            self.autonomous_agent.follow_speed_limits(True)
        elif self.args.agent == "Behavior":
            self.autonomous_agent = BehaviorAgent(self.ego_vehicle, behavior=self.args.behavior)
        else:
            raise ValueError(f"Unknown agent type: {self.args.agent}")

        if self.autonomous_agent is not None:
            # Set initial destination for the agent
            destination = random.choice(spawn_points).location
            self.autonomous_agent.set_destination(destination)
            print(f"Agent destination set to {destination}")

        # Wait a bit for the vehicle to settle
        if self.sync:
            self.world.tick()
        else:
            time.sleep(1)

    def _setup_sensors(self):
        """Set up all sensors for the ego vehicle"""
        bound_x = 2.9508416652679443  # For Lincoln MKZ 2017
        bound_y = 1.5641621351242065
        bound_z = 1.255373239517212

        # Define sensor configurations
        sensor_configs = [
            # RGB cameras
            {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'RGB_central'},

            {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': -46,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'RGB_left'},

            {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 46.0,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'RGB_right'},

            # Depth cameras
            {'type': 'sensor.camera.depth', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'DEPTH_central'},

            {'type': 'sensor.camera.depth', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': -46,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'DEPTH_left'},

            {'type': 'sensor.camera.depth', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 46.0,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'DEPTH_right'},

            # Semantic Segmentation cameras
            {'type': 'sensor.camera.semantic_segmentation', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'SS_central'},

            {'type': 'sensor.camera.semantic_segmentation', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': -46,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'SS_left'},

            {'type': 'sensor.camera.semantic_segmentation', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 46.0,
             'width': 960, 'height': 540, 'fov': 45, 'id': 'SS_right'},
        ]

        # Create all sensors
        blueprint_library = self.world.get_blueprint_library()
        for sensor_config in sensor_configs:
            # Get the blueprint for the sensor
            sensor_bp = blueprint_library.find(sensor_config['type'])

            # Set sensor attributes
            sensor_bp.set_attribute('image_size_x', str(sensor_config['width']))
            sensor_bp.set_attribute('image_size_y', str(sensor_config['height']))
            sensor_bp.set_attribute('fov', str(sensor_config['fov']))

            # Create sensor location and rotation
            sensor_location = carla.Location(x=sensor_config['x'], y=sensor_config['y'], z=sensor_config['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_config['pitch'], roll=sensor_config['roll'], yaw=sensor_config['yaw'])
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)

            # Spawn the sensor
            sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.ego_vehicle)
            self.sensors.append(sensor)

            # Set up callback to collect the data for each "camera" sensor
            if sensor_config['type'] in ['sensor.camera.rgb',
                                         'sensor.camera.depth',
                                         'sensor.camera.semantic_segmentation']:
                sensor.listen(lambda image, sensor_id=sensor_config['id']: self.camera_callback(image, sensor_id))

        # Wait for sensors to be ready
        if self.sync:
            self.world.tick()
        else:
            time.sleep(1)

    def _setup_docker(self):
        pass

    def camera_callback(self, image, sensor_id):
        """Generic callback for camera data"""
        # Store the data instead of saving immediately
        self.sensor_data[sensor_id] = image

        # Mark this sensor as having reported data for this frame
        self.sensor_data_received[sensor_id] = True

        # Check if all sensors have reported for this frame
        if len(self.sensor_data_received) >= self.num_sensors:
            self.frame_ready = True

    def _save_frame_data(self):
        """Save data for all sensors for the current frame"""
        # Map sensor types to their palettes and file extensions
        sensor_configs = {
            'RGB': {'palette': carla.ColorConverter.Raw, 'ext': 'jpg'},
            'DEPTH': {'palette': carla.ColorConverter.Depth, 'ext': 'png'},
            'SS': {'palette': carla.ColorConverter.CityScapesPalette, 'ext': 'png'}
        }

        for sensor_type, config in sensor_configs.items():
            for direction in ['left', 'central', 'right']:
                sensor_id = f'{sensor_type}_{direction}'
                if sensor_id in self.sensor_data:
                    # Create the directory if it doesn't exist
                    dir_path = os.path.join(self.output_dir, sensor_type, direction)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    # Save the image to disk
                    save_path = os.path.join(dir_path, f'{sensor_type}_{direction}_{self.frame_count:06d}.{config["ext"]}')

                    # Apply palette if specified and save
                    self.sensor_data[sensor_id].convert(config['palette'])
                    self.sensor_data[sensor_id].save_to_disk(save_path)

        # Clear the data storage after saving
        self.sensor_data = {}

    def _spawn_other_vehicles(self):
        """Spawn other vehicles in the world"""
        if self.number_of_vehicles <= 0:
            return

        print(f"Spawning {self.number_of_vehicles} vehicles...")

        # Get blueprints for vehicles
        # blueprints = get_actor_blueprints(self.world, "vehicle.*", self.args.filterv)
        blueprints = get_actor_blueprints(self.world, self.args.filterv, self.args.generationv)
        if not blueprints:
            print("No vehicle blueprints found with the specified filter")
            return

        # Filter out bicycles and motorcycles if safety option is enabled
        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        # Sort blueprints by ID
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # Get spawn points
        self.map_spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(self.map_spawn_points)

        if self.number_of_vehicles < number_of_spawn_points:
            random.shuffle(self.map_spawn_points)
        elif self.number_of_vehicles > number_of_spawn_points:
            print(f"Warning: Requested {self.number_of_vehicles} vehicles, but only {number_of_spawn_points} spawn points available")
            self.number_of_vehicles = number_of_spawn_points

        # Spawn vehicles using batch commands
        batch = []
        for n, transform in enumerate(self.map_spawn_points):
            if n >= self.number_of_vehicles:
                break

            # Skip if this is the spawn point of the ego vehicle
            if self.spawn_point_idx == n:
                continue

            blueprint = random.choice(blueprints)

            # Set random color if available
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            # Set random driver ID if available
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            # Set role name
            blueprint.set_attribute('role_name', 'autopilot')

            # Spawn the vehicles and set their autopilot
            batch.append(carla.command.SpawnActor(blueprint, transform)
                         .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.traffic_manager.get_port())))

        # Apply batch commands
        vehicle_ids = []
        for response in self.client.apply_batch_sync(batch, self.sync):
            if response.error:
                print(f"Error spawning vehicle: {response.error}")
            else:
                vehicle_ids.append(response.actor_id)

        # Set automatic vehicle lights if enabled
        if self.args.car_lights_on:
            all_vehicle_actors = self.world.get_actors(vehicle_ids)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)

        self.vehicles_list = vehicle_ids
        print(f"Spawned {len(vehicle_ids)} vehicles successfully")

    def _spawn_pedestrians(self):
        """Spawn pedestrians in the world"""
        if self.number_of_walkers <= 0:
            return

        print(f"Spawning {self.number_of_walkers} pedestrians...")

        # Get blueprints for pedestrians
        # blueprints = get_actor_blueprints(self.world, "walker.pedestrian.*", self.args.filterw)
        blueprints = get_actor_blueprints(self.world, self.args.filterw, self.args.generationw)
        if not blueprints:
            print("No pedestrian blueprints found with the specified filter")
            return

        # Set percentage of pedestrians running and crossing
        percentage_pedestrians_running = 0.2    # how many pedestrians will run
        percentage_pedestrians_crossing = 0.5   # how many pedestrians will walk through the road

        # 1. Take all the random locations to spawn
        spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 2. Spawn walker objects
        batch = []
        walker_speeds = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)

            # Set to not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # Set the max speed
            if walker_bp.has_attribute('speed'):
                if random.random() > percentage_pedestrians_running:
                    # Walking
                    walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # Running
                    walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speeds.append(0.0)

            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # Apply batch and get results
        results = self.client.apply_batch_sync(batch, True)
        walker_speeds2 = []
        walkers_list = []

        for i, result in enumerate(results):
            if result.error:
                print(f"Error spawning walker: {result.error}")
            else:
                walkers_list.append({"id": result.actor_id})
                walker_speeds2.append(walker_speeds[i])

        walker_speeds = walker_speeds2

        # 3. Spawn walker controllers
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i, walker in enumerate(walkers_list):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker["id"]))

        # Apply batch and get results
        results = self.client.apply_batch_sync(batch, True)
        for i, result in enumerate(results):
            if result.error:
                print(f"Error spawning walker controller: {result.error}")
            else:
                walkers_list[i]["con"] = result.actor_id

        # 4. Put together walker and controller IDs
        all_id = []
        for walker in walkers_list:
            all_id.append(walker["con"])
            all_id.append(walker["id"])

        self.walkers_list = all_id

        # Wait for a tick to ensure client receives the last transform of the walkers
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # 5. Initialize each controller and set target to walk to
        # Set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)

        all_actors = self.world.get_actors(all_id)
        for i in range(0, len(all_id), 2):
            # Start walker
            all_actors[i].start()
            # Set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # Set max speed
            all_actors[i].set_max_speed(float(walker_speeds[int(i/2)]))

        print(f"Spawned {len(walkers_list)} pedestrians successfully")

        # Store the controllers for later cleanup
        self.pedestrian_controllers = [all_actors[i] for i in range(0, len(all_id), 2)]

    def _simulation_loop(self):
        """Main simulation loop for data collection"""
        print(f"Waiting {self.args.settling_time} seconds for the vehicle to settle...")

        # Wait for the vehicle to settle
        settling_ticks = int(self.args.settling_time * self.args.fps) if self.sync else int(self.args.settling_time * 10)
        for _ in range(settling_ticks):
            if self.sync:
                self.world.tick()
            else:
                time.sleep(0.1)

        print("Starting data collection. Press Ctrl+C to stop.")

        pbar = None
        try:
            # Run for a specified number of frames or indefinitely if not specified
            max_frames = self.args.frames if self.args.frames > 0 else float('inf')

            # Create a progress bar if max_frames is a finite number
            if max_frames < float('inf'):
                pbar = tqdm.tqdm(total=max_frames, desc="Collecting data", unit="frames")

            while self.frame_count < max_frames:
                if self.sync:
                    # Synchronous mode

                    # If we use special agents, get the control
                    if self.autonomous_agent is not None and self.args.agent != "Autopilot":
                        control = self.autonomous_agent.run_step()
                        control.manual_gear_shift = False
                        self.ego_vehicle.apply_control(control)

                    self.world.tick()

                    # Wait until all sensors have reported (with timeout)
                    timeout = time.time() + 1.0  # 1 second timeout
                    while not self.frame_ready and time.time() < timeout:
                        time.sleep(0.01)
                        # Check if all sensors have reported
                        if len(self.sensor_data_received) >= self.num_sensors:
                            self.frame_ready = True

                    # Print progress periodically if not using a progress bar
                    if self.frame_count % 100 == 0 and pbar is None:
                        print(f"Processed {self.frame_count} frames")

                    # Only increment the frame count if we got data from all sensors
                    if self.frame_ready:
                        self._save_frame_data()  # Save data for all sensors
                        self.frame_count += 1

                        # Update progress bar if it exists
                        if pbar is not None:
                            pbar.update(1)

                        # Reset for next frame
                        self.sensor_data_received = {}
                        self.frame_ready = False

                    else:
                        print(f"Warning: Not all sensors reported for frame {self.frame_count}, got {len(self.sensor_data_received)}/{self.num_sensors}")
                        # Clear any partial data for this frame
                        self.sensor_data = {}
                        self.sensor_data_received = {}
                        self.frame_ready = False
                else:
                    # Asynchronous mode - similar but with wait_for_tick instead
                    self.world.wait_for_tick()
                    time.sleep(0.1)  # Sleep to reduce CPU usage

                    # If we use special agents, get the control
                    if self.autonomous_agent is not None and self.args.agent != "Autopilot":
                        control = self.autonomous_agent.run_step()
                        control.manual_gear_shift = False
                        self.ego_vehicle.apply_control(control)

                    # Wait until all sensors have reported (with timeout)
                    timeout = time.time() + 1.0  # 1 second timeout
                    while not self.frame_ready and time.time() < timeout:
                        time.sleep(0.01)
                        # Check if all sensors have reported
                        if len(self.sensor_data_received) >= self.num_sensors:
                            self.frame_ready = True

                    # Print progress periodically if not using a progress bar
                    if self.frame_count % 100 == 0 and pbar is None:
                        print(f"Processed {self.frame_count} frames")

                    # Only increment the frame count if we got data from all sensors
                    if self.frame_ready:
                        self._save_frame_data()  # Save data for all sensors
                        self.frame_count += 1

                        # Update progress bar if it exists
                        if pbar is not None:
                            pbar.update(1)

                        # Reset for next frame
                        self.sensor_data_received = {}
                        self.frame_ready = False
                    else:
                        print(f"Warning: Not all sensors reported for frame {self.frame_count}, got {len(self.sensor_data_received)}/{self.num_sensors}")
                        # Clear any partial data for this frame
                        self.sensor_data = {}
                        self.sensor_data_received = {}
                        self.frame_ready = False

                if self.autonomous_agent is not None:
                    if self.autonomous_agent.done():
                        self.autonomous_agent.set_destination(random.choice(self.map_spawn_points).location)
                        print("The target has been reached, searching for another target")


        except KeyboardInterrupt:
            print(f"\nData collection stopped after {self.frame_count} frames")

        finally:
            # Make sure to close the progress bar
            if pbar is not None:
                pbar.close()

    def _process_frame(self):
        """Process a single frame of data"""
        # In synchronous mode, sensor callbacks are triggered by world.tick()
        # In asynchronous mode, sensor callbacks are triggered by the simulation
        pass

    def cleanup(self):
        """Clean up all actors and sensors"""
        print("Cleaning up actors and sensors...")

        # Stop all pedestrian controllers
        for controller in self.pedestrian_controllers:
            controller.stop()

        # Destroy all sensors
        for sensor in self.sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        # Destroy pedestrians and vehicles
        print(f"Destroying {len(self.walkers_list) // 2} pedestrians")
        self.client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in self.walkers_list])

        print(f"Destroying {len(self.vehicles_list)} vehicles")
        self.client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in self.vehicles_list])

        # Destroy ego vehicle
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()

        # Reset synchronous mode settings
        if self.sync and self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        # Clear the autonomous agent if we created one
        self.autonomous_agent = None

        print("Cleanup complete")

# =============================== Helper functions ================================

def find_weather_presets():
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), x) for x in presets]


def list_options(client):
    maps = [m.replace('/Game/Carla/Maps/', '') for m in client.get_available_maps()]
    indent = 4 * ' '
    def wrap(text):
        return '\n'.join(textwrap.wrap(text, initial_indent=indent, subsequent_indent=indent))
    print('weather presets:\n')
    print(wrap(', '.join(x for _, x in find_weather_presets())) + '.\n')
    print('available maps:\n')
    print(wrap(', '.join(sorted(maps))) + '.\n')

# ================================= Main function =================================

def main():
    """Parse arguments and start the data collector"""
    argparser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    argparser.add_argument('-d', '--docker', metavar='DOCKER_IMAGE', default=None, help='Docker image to use (default: %(default)s, recommended: "carlasim/carla:0.9.15")')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: %(default)s)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: %(default)s)')
    argparser.add_argument('-nr', '--no-render', action='store_true', default=False, help='Do not render the simulation (default: %(default)s)')
    argparser.add_argument('-m', '--map', default='Town15', help='Load a new map (Town01-Town15, or full path; default: %(default)s)')
    argparser.add_argument('-we', '--weather', default=None, help='Weather preset (default: %(default)s)')
    argparser.add_argument('-l', '--list', action='store_true', default=False, help='List available maps and weathers and exit')
    argparser.add_argument('-o', '--output-dir', default='_out', help='Output directory for collected data', required=True)
    argparser.add_argument('-n', '--number-of-vehicles', default=30, type=int, help='Number of vehicles (default: %(default)s)')
    argparser.add_argument('-w', '--number-of-walkers', default=10, type=int, help='Number of walkers (default: %(default)s)')
    argparser.add_argument('-s', '--spawn-point', default=-1, type=int, help='Spawn point index for ego vehicle; if  (default: %(default)s)')
    argparser.add_argument('--filterv', metavar='PATTERN', default='vehicle.*', help='Filter vehicle models (default: %(default)s)')
    argparser.add_argument('--filterw', metavar='PATTERN', default='walker.pedestrian.*', help='Filter pedestrian models (default: %(default)s)')
    argparser.add_argument('--generationv', metavar='G', default='All', help='Vehicle generation (values: "1","2","All" - default: %(default)s)')
    argparser.add_argument('--generationw', metavar='G', default='2', help='Pedestrian generation (values: "1","2","All" - default: %(default)s)')
    argparser.add_argument("--agent", type=str, choices=["Behavior", "Basic", "Constant", "Autopilot"], help="select which agent to run (default: %(default)s)", default="Autopilot")
    argparser.add_argument('--behavior', type=str, choices=["cautious", "normal", "aggressive"], help='Choose agent behavior (for Behavior agent only, default: %(default)s)', default='normal')
    argparser.add_argument('--max-speed', type=float, default=30.0, help='Max speed for agents, not including Autopilot (default: %(default)s)')
    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port for Traffic Manager (default: %(default)s)')
    argparser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    argparser.add_argument('--seed', type=int, default=0, help='Set a seed for the random number generator (default: %(default)s)')
    argparser.add_argument('--car-lights-on', action='store_true', default=False, help='Enable car lights')
    argparser.add_argument('--safe', action='store_true', default=False, help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument('--frames', type=int, default=0, help='Number of frames to collect; 0 for infinite (default: %(default)s)')
    argparser.add_argument('--fps', type=float, default=20.0, help='Frames per second for synchronous mode (default: %(default)s)')
    argparser.add_argument('--data-fps', type=float, default=20.0, help='Data collection frequency (default: %(default)s)')
    argparser.add_argument('--settling-time', type=float, default=1.0, help='Time in seconds to wait for the vehicle to settle before starting data collection (default: %(default)s)')

    args = argparser.parse_args()

    # Configure logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # List available maps and weathers if requested and exit
    if args.list:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        list_options(client)
        return

    try:
        collector = CarlaSensorDataCollector(args)
        collector.start()
    except KeyboardInterrupt:
        print('\nCancelled by user.')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        print('\nDone.')

if __name__ == '__main__':
    main()