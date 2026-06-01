
//
//  MainViewModel.swift
//  LNNDemo
//

import Foundation
import Combine

class MainViewModel: ObservableObject {
    @Published var timeSeriesData: [(time: Double, value: Double)] = []
    @Published var predictions: [(time: Double, value: Double)] = []
    @Published var isRunning: Bool = false
    @Published var currentPrediction: Double = 0.0
    @Published var currentInput: Double = 0.0
    
    private let maxDataPoints: Int = 100
    private var timeStep: Double = 0.0
    private var timer: Timer?
    private let predictor = SimpleSinePredictor()
    
    init() {
        resetData()
    }
    
    func resetData() {
        timeSeriesData.removeAll()
        predictions.removeAll()
        timeStep = 0.0
        currentInput = 0.0
        currentPrediction = 0.0
        predictor.reset()
        
        for i in 0..<20 {
            let t = Double(i) * 0.1
            let value = sin(2.0 * Double.pi * 0.1 * t)
            timeSeriesData.append((t, value))
            timeStep = t
        }
    }
    
    func startSimulation() {
        guard !isRunning else { return }
        isRunning = true
        
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateSimulation()
        }
    }
    
    func stopSimulation() {
        isRunning = false
        timer?.invalidate()
        timer = nil
    }
    
    private func updateSimulation() {
        timeStep += 0.1
        
        let trueValue = sin(2.0 * Double.pi * 0.1 * timeStep)
        currentInput = trueValue
        
        let prediction = predictor.updateAndPredict(Float(trueValue))
        currentPrediction = Double(prediction)
        
        timeSeriesData.append((timeStep, trueValue))
        predictions.append((timeStep, currentPrediction))
        
        if timeSeriesData.count > maxDataPoints {
            timeSeriesData.removeFirst()
            predictions.removeFirst()
        }
    }
    
    func generateCustomData(type: DataGeneratorType) {
        stopSimulation()
        resetData()
        timeSeriesData.removeAll()
        
        let count = 100
        for i in 0..<count {
            let t = Double(i) * 0.1
            let value: Double
            
            switch type {
            case .sine:
                value = sin(2.0 * Double.pi * 0.1 * t)
            case .doubleSine:
                value = sin(2.0 * Double.pi * 0.1 * t) + 0.5 * sin(2.0 * Double.pi * 0.3 * t)
            case .noisySine:
                value = sin(2.0 * Double.pi * 0.1 * t) + Double.random(in: -0.1...0.1)
            case .step:
                value = t < 5.0 ? 0.0 : 1.0
            }
            
            timeSeriesData.append((t, value))
            timeStep = t
        }
    }
}

enum DataGeneratorType {
    case sine
    case doubleSine
    case noisySine
    case step
}

