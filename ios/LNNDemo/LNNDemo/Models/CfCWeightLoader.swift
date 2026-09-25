//
//  CfCWeightLoader.swift
//  LNNDemo
//
//  Loads trained CfC weights from a JSON file produced by
//  scripts/export_lnn_for_ios.py.
//
//  Pre-CfCWeightLoader the Swift app hard-coded `0.1` weights in
//  `CfCModel.createPreTrained()` (placeholder) — the trained .pt from
//  export was never loaded at runtime. This loader bridges the gap:
//  `loadFromBundle(name:fallback:)` looks for `<name>.json` in the main
//  bundle; if absent, returns `fallback` so the demo still runs.
//
//  JSON schema expected:
//
//    {
//      "hidden_size": 8,
//      "f_gate_weight": [[..], ..],   // shape: [in_dim + hidden, hidden]
//      "f_gate_bias":   [..],          // shape: [hidden]
//      "g_branch_weight": [[..], ..],
//      "g_branch_bias":   [..],
//      "h_branch_weight": [[..], ..],
//      "h_branch_bias":   [..],
//      "time_scale":      [..],
//      "output_weight":   [[..], ..],  // shape: [hidden, out_dim]
//      "output_bias":     [..]         // shape: [out_dim]
//    }

import Foundation

enum CfCWeightLoaderError: Error {
    case missingResource(String)
    case malformedJSON(String)
    case shapeMismatch(String)
}

struct CfCWeights {
    let hiddenSize: Int
    let fGateWeight: [[Float]]
    let fGateBias: [Float]
    let gBranchWeight: [[Float]]
    let gBranchBias: [Float]
    let hBranchWeight: [[Float]]
    let hBranchBias: [Float]
    let timeScale: [Float]
    let outputWeight: [[Float]]
    let outputBias: [Float]
}

enum CfCWeightLoader {
    /// Try to load trained weights from a bundle resource. Returns `nil`
    /// if the resource is missing — caller should then use placeholder weights.
    static func loadFromBundle(name: String = "cfc_weights") -> CfCWeights? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "json") else {
            return nil
        }
        do {
            return try load(from: url)
        } catch {
            print("[CfCWeightLoader] failed to parse \(name).json: \(error)")
            return nil
        }
    }

    static func load(from url: URL) throws -> CfCWeights {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CfCWeightLoaderError.malformedJSON("root is not an object")
        }
        guard let hidden = dict["hidden_size"] as? Int else {
            throw CfCWeightLoaderError.malformedJSON("missing hidden_size (int)")
        }
        let fGW = try floatMatrix(dict["f_gate_weight"], expectedRows: hidden + 1, cols: hidden, name: "f_gate_weight")
        let fGB = try floatArray(dict["f_gate_bias"], expectedLen: hidden, name: "f_gate_bias")
        let gBW = try floatMatrix(dict["g_branch_weight"], expectedRows: hidden + 1, cols: hidden, name: "g_branch_weight")
        let gBB = try floatArray(dict["g_branch_bias"], expectedLen: hidden, name: "g_branch_bias")
        let hBW = try floatMatrix(dict["h_branch_weight"], expectedRows: hidden + 1, cols: hidden, name: "h_branch_weight")
        let hBB = try floatArray(dict["h_branch_bias"], expectedLen: hidden, name: "h_branch_bias")
        let ts = try floatArray(dict["time_scale"], expectedLen: hidden, name: "time_scale")
        let oW = try floatMatrix(dict["output_weight"], expectedRows: hidden, cols: 1, name: "output_weight")
        let oB = try floatArray(dict["output_bias"], expectedLen: 1, name: "output_bias")
        return CfCWeights(
            hiddenSize: hidden,
            fGateWeight: fGW, fGateBias: fGB,
            gBranchWeight: gBW, gBranchBias: gBB,
            hBranchWeight: hBW, hBranchBias: hBB,
            timeScale: ts,
            outputWeight: oW, outputBias: oB
        )
    }

    // MARK: - Helpers

    private static func floatArray(_ raw: Any?, expectedLen: Int, name: String) throws -> [Float] {
        guard let arr = raw as? [Any] else {
            throw CfCWeightLoaderError.malformedJSON("\(name) is not an array")
        }
        var out: [Float] = []
        out.reserveCapacity(arr.count)
        for (i, x) in arr.enumerated() {
            guard let v = (x as? NSNumber)?.floatValue else {
                throw CfCWeightLoaderError.malformedJSON("\(name)[\(i)] not a number")
            }
            out.append(v)
        }
        if out.count != expectedLen {
            throw CfCWeightLoaderError.shapeMismatch("\(name) expected \(expectedLen), got \(out.count)")
        }
        return out
    }

    private static func floatMatrix(_ raw: Any?, expectedRows: Int, cols: Int, name: String) throws -> [[Float]] {
        guard let arr = raw as? [Any] else {
            throw CfCWeightLoaderError.malformedJSON("\(name) is not an array")
        }
        if arr.count != expectedRows {
            throw CfCWeightLoaderError.shapeMismatch("\(name) expected \(expectedRows) rows, got \(arr.count)")
        }
        var out: [[Float]] = []
        out.reserveCapacity(expectedRows)
        for (i, row) in arr.enumerated() {
            guard let rowArr = row as? [Any] else {
                throw CfCWeightLoaderError.malformedJSON("\(name)[\(i)] not an array")
            }
            var rowOut: [Float] = []
            rowOut.reserveCapacity(cols)
            for (j, x) in rowArr.enumerated() {
                guard let v = (x as? NSNumber)?.floatValue else {
                    throw CfCWeightLoaderError.malformedJSON("\(name)[\(i)][\(j)] not a number")
                }
                rowOut.append(v)
            }
            if rowOut.count != cols {
                throw CfCWeightLoaderError.shapeMismatch(
                    "\(name)[\(i)] expected \(cols) cols, got \(rowOut.count)"
                )
            }
            out.append(rowOut)
        }
        return out
    }
}