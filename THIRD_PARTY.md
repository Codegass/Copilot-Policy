# Third-party components

The optional hardware backends use these upstream projects:

- [ARX5 SDK](https://github.com/real-stanford/arx5-sdk), MIT License. Install with `pip install -e .[arx]` or `python scripts/install_drivers.py arx`.
- [I2RT](https://github.com/i2rt-robotics/i2rt), upstream revision `ac096928d6899ddf852a71c5e8fbaa6055cd9745`. Install only for YAM with `python scripts/install_drivers.py yam`; review its repository license and dependencies.
- Intel RealSense Python bindings, installed with `python scripts/install_drivers.py realsense`.

The public package does not vendor these SDKs. Check each upstream project's terms before redistribution.
