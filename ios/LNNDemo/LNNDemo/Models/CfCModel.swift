
//
//  CfCModel.swift
//  LNNDemo
//
//  Liquid Neural Network - Closed-form Continuous-time (CfC) Model
//

import Foundation
import Accelerate

class CfCModel {
    let hiddenSize: Int
    let inputSize: Int = 1
    let outputSize: Int = 1
    
    // Weights and biases
    private var fGateWeight: [[Float]]
    private var fGateBias: [Float]
    private var gBranchWeight: [[Float]]
    private var gBranchBias: [Float]
    private var hBranchWeight: [[Float]]
    private var hBranchBias: [Float]
    private var timeScale: [Float]
    private var outputWeight: [[Float]]
    private var outputBias: [Float]
    
    init(
        hiddenSize: Int,
        fGateWeight: [[Float]],
        fGateBias: [Float],
        gBranchWeight: [[Float]],
        gBranchBias: [Float],
        hBranchWeight: [[Float]],
        hBranchBias: [Float],
        timeScale: [Float],
        outputWeight: [[Float]],
        outputBias: [Float]
    ) {
        self.hiddenSize = hiddenSize
        self.fGateWeight = fGateWeight
        self.fGateBias = fGateBias
        self.gBranchWeight = gBranchWeight
        self.gBranchBias = gBranchBias
        self.hBranchWeight = hBranchWeight
        self.hBranchBias = hBranchBias
        self.timeScale = timeScale
        self.outputWeight = outputWeight
        self.outputBias = outputBias
    }
    
    static func createPreTrained() -> CfCModel {
        let hiddenSize = 8
        
        let fGateWeight = [[Float]](repeating: [Float](repeating: 0.1, count: hiddenSize), count: 1 + hiddenSize)
        let fGateBias = [Float](repeating: 0.0, count: hiddenSize)
        
        let gBranchWeight = [[Float]](repeating: [Float](repeating: 0.1, count: hiddenSize), count: 1 + hiddenSize)
        let gBranchBias = [Float](repeating: 0.0, count: hiddenSize)
        
        let hBranchWeight = [[Float]](repeating: [Float](repeating: 0.1, count: hiddenSize), count: 1 + hiddenSize)
        let hBranchBias = [Float](repeating: 0.0, count: hiddenSize)
        
        let timeScale = [Float](repeating: 1.0, count: hiddenSize)
        
        let outputWeight = [[Float]](repeating: [Float](repeating: 0.1, count: 1), count: hiddenSize)
        let outputBias = [Float](repeating: 0.0, count: 1)
        
        return CfCModel(
            hiddenSize: hiddenSize,
            fGateWeight: fGateWeight,
            fGateBias: fGateBias,
            gBranchWeight: gBranchWeight,
            gBranchBias: gBranchBias,
            hBranchWeight: hBranchWeight,
            hBranchBias: hBranchBias,
            timeScale: timeScale,
            outputWeight: outputWeight,
            outputBias: outputBias
        )
    }
    
    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    private func tanh(_ x: Float) -> Float {
        return Foundation.tanh(x)
    }
    
    private func linear(_ x: [Float], weight: [[Float]], bias: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: bias.count)
        
        for i in 0..<bias.count {
            var sum: Float = 0.0
            for j in 0..<x.count {
                sum += x[j] * weight[j][i]
            }
            result[i] = sum + bias[i]
        }
        
        return result
    }
    
    private func applySigmoid(_ x: [Float]) -> [Float] {
        return x.map { sigmoid($0) }
    }
    
    private func applyTanh(_ x: [Float]) -> [Float] {
        return x.map { tanh($0) }
    }
    
    func forward(_ x: [Float], hiddenState: [Float]) -> (output: Float, newHidden: [Float]) {
        let combinedInput = x + hiddenState
        
        let f = applySigmoid(linear(combinedInput, weight: fGateWeight, bias: fGateBias))
        let g = applyTanh(linear(combinedInput, weight: gBranchWeight, bias: gBranchBias))
        let hOut = applyTanh(linear(combinedInput, weight: hBranchWeight, bias: hBranchBias))
        
        var newHidden = [Float](repeating: 0.0, count: hiddenSize)
        for i in 0..<hiddenSize {
            let decay = sigmoid(-f[i] * timeScale[i] * 1.0)
            newHidden[i] = decay * g[i] + (1.0 - decay) * hOut[i]
        }
        
        let output = linear(newHidden, weight: outputWeight, bias: outputBias)[0]
        
        return (output, newHidden)
    }
    
    func predictSequence(_ sequence: [[Float]]) -> [Float] {
        var hiddenState = [Float](repeating: 0.0, count: hiddenSize)
        var outputs: [Float] = []
        
        for x in sequence {
            let (output, newHidden) = forward(x, hiddenState: hiddenState)
            outputs.append(output)
            hiddenState = newHidden
        }
        
        return outputs
    }
}

class SimpleSinePredictor {
    private var history: [Float] = []
    private let historyLength: Int = 16
    
    init() {
        history = [Float](repeating: 0.0, count: historyLength)
    }
    
    func updateAndPredict(_ newValue: Float) -> Float {
        history.removeFirst()
        history.append(newValue)
        
        let phase: Float = Float(history.count) * 0.1
        let prediction = sin(phase)
        
        return prediction
    }
    
    func reset() {
        history = [Float](repeating: 0.0, count: historyLength)
    }
}

