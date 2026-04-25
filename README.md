# Nora Optimizer

Just import and use it as a drop-in replacement for Adam!

```python
from nora import Nora

# That's it! It automatically applies Nora to 2D parameters and Adam to 1D parameters.
optimizer = Nora(model.parameters(), lr=1e-3, weight_decay=0.0)
```
<img width="1024" height="1536" alt="729caac7349f42c86d31371fac872a26" src="https://github.com/user-attachments/assets/7723d1a4-3737-40f2-afa4-96b164477b9d" />
