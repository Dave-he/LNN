
//
//  LineChartView.swift
//  LNNDemo
//

import SwiftUI

struct LineChartView: View {
    let data: [(time: Double, value: Double)]
    let predictions: [(time: Double, value: Double)]
    let title: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)
            
            GeometryReader { geometry in
                ZStack {
                    drawGrid(in: geometry)
                    
                    if !data.isEmpty {
                        drawLine(
                            data: data,
                            in: geometry,
                            color: .blue,
                            lineWidth: 2
                        )
                    }
                    
                    if !predictions.isEmpty {
                        drawLine(
                            data: predictions,
                            in: geometry,
                            color: .orange,
                            lineWidth: 2,
                            dashed: true
                        )
                    }
                    
                    drawAxes(in: geometry)
                }
            }
            .frame(height: 250)
            .padding()
        }
    }
    
    private func drawGrid(in geometry: GeometryProxy) -&gt; some View {
        let width = geometry.size.width
        let height = geometry.size.height
        
        return Path { path in
            for i in 0...4 {
                let y = height * CGFloat(i) / 4
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: width, y: y))
            }
            
            for i in 0...4 {
                let x = width * CGFloat(i) / 4
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: height))
            }
        }
        .stroke(Color.gray.opacity(0.3), lineWidth: 0.5)
    }
    
    private func drawAxes(in geometry: GeometryProxy) -&gt; some View {
        let width = geometry.size.width
        let height = geometry.size.height
        
        return Path { path in
            path.move(to: CGPoint(x: 0, y: height/2))
            path.addLine(to: CGPoint(x: width, y: height/2))
        }
        .stroke(Color.gray, lineWidth: 1)
    }
    
    private func drawLine(
        data: [(time: Double, value: Double)],
        in geometry: GeometryProxy,
        color: Color,
        lineWidth: CGFloat,
        dashed: Bool = false
    ) -&gt; some View {
        let width = geometry.size.width
        let height = geometry.size.height
        
        guard let minTime = data.first?.time, let maxTime = data.last?.time else {
            return AnyView(EmptyView())
        }
        
        let timeRange = maxTime - minTime &gt; 0 ? maxTime - minTime : 1
        
        let values = data.map { $0.value }
        let minValue = values.min() ?? -1
        let maxValue = values.max() ?? 1
        let valueRange = maxValue - minValue &gt; 0 ? maxValue - minValue : 2
        
        return Path { path in
            for (index, point) in data.enumerated() {
                let x = width * CGFloat((point.time - minTime) / timeRange)
                let normalizedValue = (point.value - minValue) / valueRange
                let y = height * (1 - CGFloat(normalizedValue))
                
                if index == 0 {
                    path.move(to: CGPoint(x: x, y: y))
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
        }
        .stroke(
            color,
            style: StrokeStyle(
                lineWidth: lineWidth,
                lineCap: .round,
                lineJoin: .round,
                dash: dashed ? [5, 5] : []
            )
        )
    }
}

struct LineChartView_Previews: PreviewProvider {
    static var previews: some View {
        let data = (0..&lt;50).map { i in
            (Double(i) * 0.1, sin(Double(i) * 0.2))
        }
        let predictions = (0..&lt;50).map { i in
            (Double(i) * 0.1, cos(Double(i) * 0.2))
        }
        
        LineChartView(data: data, predictions: predictions, title: "Test Chart")
    }
}
