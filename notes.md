- Although this is not strictly necessary, we find it useful to define the model parameters in one config file (e.g configs/125M.yml) and the data path parameters in another (e.g configs/local_setup.yml).
- For most uses we recommend deploying models trained using the GPT-NeoX library via the Hugging Face Transformers library which is better optimized for inference.
  Benchmarking
  - https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token
  - https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line
  - The Py150 dataset contains 150,000 Python files.
- While so far there has been no systematic work that focuses on prompted pretraining, recent work (Biderman and Raff, 2022) observed that the formulation of the StackExchange component of the Pile appears to heavily influences code generation
- char_level_ppl for char tokenizer?
- 1.2GB so could increase batch by at least 10x
- The deepy.py script assumes it is running in the root of GTP-NeoX repo
