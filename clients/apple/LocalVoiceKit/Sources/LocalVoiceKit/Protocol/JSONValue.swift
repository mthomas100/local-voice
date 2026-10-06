import Foundation

/// A JSON value with ordered object keys.
///
/// Protocol v1 control messages are flat JSON objects with a `"t"` field (PROTOCOL.md). Keeping key order means the
/// wire text reads `{"t":"hello","v":1,...}` in logs, and a small value type lets the decoder be tolerant (a field may
/// arrive as a string or a number) without Codable boilerplate for every message.
public enum JSONValue: Sendable, Equatable, CustomStringConvertible {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([(String, JSONValue)])

    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case let (.string(a), .string(b)): return a == b
        case let (.int(a), .int(b)): return a == b
        case let (.double(a), .double(b)): return a == b
        case let (.int(a), .double(b)), let (.double(b), .int(a)): return Double(a) == b
        case let (.bool(a), .bool(b)): return a == b
        case (.null, .null): return true
        case let (.array(a), .array(b)): return a == b
        case let (.object(a), .object(b)):
            // Key order is presentation only; objects compare as dictionaries.
            guard a.count == b.count else { return false }
            let bd = Dictionary(b, uniquingKeysWith: { first, _ in first })
            return a.allSatisfy { key, value in bd[key] == value }
        default: return false
        }
    }

    public subscript(key: String) -> JSONValue? {
        guard case let .object(pairs) = self else { return nil }
        return pairs.first(where: { $0.0 == key })?.1
    }

    public var stringValue: String? {
        if case let .string(s) = self { return s }
        return nil
    }

    /// Integers, and doubles that are whole numbers (a server may send `1234.0` for milliseconds).
    public var intValue: Int? {
        switch self {
        case let .int(i): return i
        case let .double(d) where d.rounded() == d && abs(d) < 9.0e15: return Int(d)
        default: return nil
        }
    }

    /// A number, whether it came as an integer or not (the orchestrator's millisecond figures are floats).
    public var doubleValue: Double? {
        switch self {
        case let .double(d): return d
        case let .int(i): return Double(i)
        default: return nil
        }
    }

    public var arrayValue: [JSONValue]? {
        if case let .array(a) = self { return a }
        return nil
    }

    public var objectPairs: [(String, JSONValue)]? {
        if case let .object(o) = self { return o }
        return nil
    }

    public var boolValue: Bool? {
        if case let .bool(b) = self { return b }
        return nil
    }

    public var description: String { serialized() }

    // MARK: Serialization

    public func serialized() -> String {
        var out = ""
        write(into: &out)
        return out
    }

    private func write(into out: inout String) {
        switch self {
        case let .string(s): JSONValue.writeString(s, into: &out)
        case let .int(i): out += String(i)
        case let .double(d):
            if d.isFinite {
                out += d.rounded() == d && abs(d) < 1e15 ? String(Int(d)) : String(d)
            } else {
                out += "null"  // JSON has no NaN or infinity
            }
        case let .bool(b): out += b ? "true" : "false"
        case .null: out += "null"
        case let .array(items):
            out += "["
            for (i, item) in items.enumerated() {
                if i > 0 { out += "," }
                item.write(into: &out)
            }
            out += "]"
        case let .object(pairs):
            out += "{"
            for (i, (key, value)) in pairs.enumerated() {
                if i > 0 { out += "," }
                JSONValue.writeString(key, into: &out)
                out += ":"
                value.write(into: &out)
            }
            out += "}"
        }
    }

    private static func writeString(_ s: String, into out: inout String) {
        out += "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }

    // MARK: Parsing

    public enum ParseError: Error, Equatable {
        case notJSON
    }

    public static func parse(_ text: String) throws -> JSONValue {
        guard let data = text.data(using: .utf8) else { throw ParseError.notJSON }
        return try parse(data)
    }

    public static func parse(_ data: Data) throws -> JSONValue {
        let raw: Any
        do {
            raw = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        } catch {
            throw ParseError.notJSON
        }
        return convert(raw)
    }

    private static func convert(_ raw: Any) -> JSONValue {
        switch raw {
        case let s as String:
            return .string(s)
        case let n as NSNumber:
            // JSONSerialization returns NSNumber for booleans too; CFBoolean is the only way to tell them apart.
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return .bool(n.boolValue) }
            if CFNumberIsFloatType(n) { return .double(n.doubleValue) }
            return .int(n.intValue)
        case let a as [Any]:
            return .array(a.map(convert))
        case let d as [String: Any]:
            return .object(d.keys.sorted().map { ($0, convert(d[$0]!)) })
        default:
            return .null
        }
    }
}

extension JSONValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByBooleanLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}
