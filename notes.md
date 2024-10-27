- Although this is not strictly necessary, we find it useful to define the model parameters in one config file (e.g configs/125M.yml) and the data path parameters in another (e.g configs/local_setup.yml).
- For most uses we recommend deploying models trained using the GPT-NeoX library via the Hugging Face Transformers library which is better optimized for inference.
Benchmarking
    - https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token
    - https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line
    - The Py150 dataset contains 150,000 Python files.
