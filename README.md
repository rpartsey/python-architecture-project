# python-architecture-project
Python Architecture Project consists of three services: 
- `streamlit-ui`: web iterface for interaction
- `ml-api`: service that provised api to evaluate the trained Burned Areas Segmentation model on 4-channel [Planet](https://www.planet.com/) satellite images
- `metrics-api`: service that provised api to calculate IoU score (and, potentially other metrics)

Execute `docker-compose up` to run the project.

Test images and ground truth mask can be found in `images` folder.

See demo videos in `demo` folder.
