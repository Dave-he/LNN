
# LNN iOS Demo

A SwiftUI-based iOS application that demonstrates Liquid Neural Networks (LNN), specifically Closed-form Continuous-time (CfC) networks, for time-series prediction on iPad and iPhone.

## Features

- **Real-time simulation**: Watch the LNN predict sine wave patterns
- **Multiple data generators**: Test with different time-series patterns
- **Interactive controls**: Start/stop, reset, and switch between data types
- **Beautiful charts**: Visualize true values vs. predictions
- **Lightweight implementation**: Pure Swift, no external dependencies

## Getting Started

### Requirements

- Xcode 14.0+
- iOS 16.0+
- Swift 5.7+

### Building the Project

1. Open Xcode
2. Open the `LNNDemo` project
3. Select your target device (iPad recommended)
4. Press `Cmd + R` to build and run

### Project Structure

```
LNNDemo/
├── LNNDemo/
│   ├── Models/
│   │   └── CfCModel.swift       # CfC neural network implementation
│   ├── ViewModels/
│   │   └── MainViewModel.swift  # Business logic and state management
│   ├── Views/
│   │   ├── ContentView.swift    # Main UI
│   │   └── LineChartView.swift  # Chart visualization
│   └── LNNDemoApp.swift         # App entry point
└── README.md
```

## About Liquid Neural Networks

Liquid Neural Networks (LNNs) are a class of neural networks designed for sequential data and time-series prediction. Key features include:

- **Continuous-time dynamics**: Model temporal evolution continuously
- **Closed-form solutions (CfC)**: Avoid expensive ODE solvers
- **Compact size**: Fewer parameters than traditional RNNs/LSTMs
- **Interpretability**: Clearer understanding of internal dynamics

## Next Steps

1. **Train real weights**: Use the Python scripts in the parent directory to train CfC models
2. **Export weights**: Convert trained PyTorch weights to Swift format
3. **Enhance visualization**: Add more chart types and metrics
4. **Add real data**: Integrate sensor data or other real-world time-series

## Troubleshooting

If you encounter issues:
- Make sure you're using a recent Xcode version
- Clean the build folder (`Cmd + Shift + K`)
- Check iOS deployment target in project settings

## License

This project is part of the LNN research repository and follows the same licensing.
