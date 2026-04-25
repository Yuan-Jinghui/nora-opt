# Nora Optimizer

Just import and use it as a drop-in replacement for Adam!

```python
from nora import Nora

# That's it! It automatically applies Nora to 2D parameters and Adam to 1D parameters.
optimizer = Nora(model.parameters(), lr=1e-3, weight_decay=0.0)
```