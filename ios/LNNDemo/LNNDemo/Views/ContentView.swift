
//
//  ContentView.swift
//  LNNDemo
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = MainViewModel()
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    headerSection
                    
                    controlSection
                    
                    chartSection
                    
                    infoSection
                }
                .padding()
            }
            .navigationTitle("Liquid Neural Network")
            .navigationBarTitleDisplayMode(.large)
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Closed-form Continuous-time (CfC)")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Text("A lightweight liquid neural network for time-series prediction on iOS devices.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var controlSection: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                Button(action: {
                    if viewModel.isRunning {
                        viewModel.stopSimulation()
                    } else {
                        viewModel.startSimulation()
                    }
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: viewModel.isRunning ? "stop.fill" : "play.fill")
                        Text(viewModel.isRunning ? "Stop" : "Start")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isRunning ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                
                Button(action: {
                    viewModel.resetData()
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: "arrow.clockwise")
                        Text("Reset")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
            }
            
            VStack(spacing: 12) {
                Text("Data Generator")
                    .font(.headline)
                    .frame(maxWidth: .infinity, alignment: .leading)
                
                HStack(spacing: 12) {
                    DataButton(title: "Sine", color: .purple) {
                        viewModel.generateCustomData(type: .sine)
                    }
                    
                    DataButton(title: "Dual", color: .orange) {
                        viewModel.generateCustomData(type: .doubleSine)
                    }
                    
                    DataButton(title: "Noisy", color: .pink) {
                        viewModel.generateCustomData(type: .noisySine)
                    }
                    
                    DataButton(title: "Step", color: .teal) {
                        viewModel.generateCustomData(type: .step)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var chartSection: some View {
        VStack(spacing: 16) {
            LineChartView(
                data: viewModel.timeSeriesData,
                predictions: viewModel.predictions,
                title: "Time Series Prediction"
            )
            
            HStack(spacing: 20) {
                LegendItem(color: .blue, label: "True Value")
                LegendItem(color: .orange, label: "Prediction", dashed: true)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var infoSection: some View {
        HStack(spacing: 20) {
            InfoCard(
                title: "Current Input",
                value: String(format: "%.4f", viewModel.currentInput),
                color: .blue
            )
            
            InfoCard(
                title: "Prediction",
                value: String(format: "%.4f", viewModel.currentPrediction),
                color: .orange
            )
        }
    }
}

struct DataButton: View {
    let title: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(color.opacity(0.2))
                .foregroundColor(color)
                .cornerRadius(8)
        }
    }
}

struct LegendItem: View {
    let color: Color
    let label: String
    var dashed: Bool = false
    
    var body: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 20, height: 3)
                .overlay(
                    dashed ?
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(Color.white, style: StrokeStyle(lineWidth: 1, dash: [2, 2]))
                    : nil
                )
            
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct InfoCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.system(.title2, design: .monospaced))
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(color.opacity(0.3), lineWidth: 1)
        )
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

