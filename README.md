# Data Collection with CARLA

The [CARLA](https://carla.org/) simulator is a leading platform for autonomous driving research based on the Unreal Engine 4.x 
and, most recently, [5.x](https://carla.org/2024/12/19/release-0.10.0/). It offers a realistic virtual environment for testing 
and developing autonomous systems. One of the key features of CARLA is its data collection functionality, which allows users to 
easily and transparently collect data during simulation runs.

This repository provides an overview of the CARLA simulator and its data collection capabilities in a clear and straightforward
way. We want to make it as easy as possible to get started in using this exciting simulator for many different experiments in 
Computer Vision, not only in autonomous driving. As such, we will provide starting code to get used to obtaining data, as well 
as adding more complex scenarios, specifically aimed for more rigorous testing of autonomous driving systems.

## Checklist

* [ ] Ensure environment can be created with the provided `requirements.txt` file
* [ ] Set up a simple
* [ ] Complete README

> [!NOTE]
> 

# Getting started

## Code and CARLA installation

```bash
# Clone this repository
git clone https://github.com/PDillis/carla-data-collection
cd carla-data-collection

# Create the environment and install required packages
conda create -n carladc python=3.10
conda activate carladc
pip3 install -r requirements.txt
```

This code will require for CARLA to be installed locally. This can be done either by downloading the respective package
version you wish to use in the simulator's [releases page](https://github.com/carla-simulator/carla/releases), or you can
also save some headache by downloading the Docker image. To do this, simply do:

```bash
docker pull carlasim/carla:0.9.15  # For the latest Unreal 4.x version
docker pull carlasim/carla:0.10.0  # For the latest Unreal 5.x version
```

