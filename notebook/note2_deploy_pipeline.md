## Obstacles

- Variables (input features) are provided in inference spec yaml file, which makes it difficult to use these variables for model inference. Variables should be provided at runtime API request, to increase maintainability and avoid server's heavy load.

## Improvement

- Create separate repository: (1) data processing and model training, and (2) ML pipeline deployment

## Disclaimer

- I do not add LICENSE as I am not sure about the copyright of this repository