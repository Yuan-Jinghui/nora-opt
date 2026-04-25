# Nora Optimizer

Just import and use it as a drop-in replacement for Adam!

```python
from nora import Nora

# That's it! It automatically applies Nora to 2D parameters and Adam to 1D parameters.
optimizer = Nora(model.parameters(), lr=1e-3, weight_decay=0.0)
```
<img width="1024" height="1536" alt="936323f19d90b00c6f1a2f67e55aa0a4" src="https://github.com/user-attachments/assets/960390e8-5118-4242-b247-46561c02d865" />
