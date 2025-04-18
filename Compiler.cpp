// Lustra Compiler (Prototype) - Written in C++
// Parses Lustra syntax and generates executable C++ code

#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

class LustraCompiler {
public:
    std::string compile(const std::string& source) {
        std::istringstream iss(source);
        std::string line;
        std::string compiled;
        int indent = 0;
        bool inFunction = false;

        while (std::getline(iss, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;

            auto [compiledLine, isFunctionStart, isFunctionEnd] = compileLine(trimmed, inFunction);

            if (isFunctionEnd && indent > 0) indent--;

            compiled += std::string(indent * 4, ' ') + compiledLine + "\n";

            if (isFunctionStart) {
                indent++;
                inFunction = true;
            }
            if (isFunctionEnd) inFunction = false;
        }

        return wrapWithMain(compiled);
    }

private:
    std::string mapType(const std::string& ltype) {
        if (ltype == "String") return "std::string";
        if (ltype == "Int") return "int";
        if (ltype == "Float") return "float";
        if (ltype == "Bool") return "bool";
        return "auto";
    }

    std::tuple<std::string, bool, bool> compileLine(const std::string& line, bool inFunction) {
        std::smatch match;

        // Variable with type
        if (std::regex_match(line, match, std::regex(R"(let (\w+): (\w+) = (.+))"))) {
            std::string name = match[1];
            std::string ltype = match[2];
            std::string value = match[3];
            return {mapType(ltype) + " " + name + " = " + value + ";", false, false};
        }

        // Variable inferred
        if (std::regex_match(line, match, std::regex(R"(let (\w+) = (.+))"))) {
            std::string name = match[1];
            std::string value = match[2];
            return {"auto " + name + " = " + value + ";", false, false};
        }

        // Print
        if (std::regex_match(line, match, std::regex(R"(print\((.+)\))"))) {
            return {"std::cout << " + match[1].str() + " << std::endl;", false, false};
        }

        // Function definition
        if (std::regex_match(line, match, std::regex(R"(func (\w+)\((.*?)\): (\w+) =>)"))) {
            std::string name = match[1];
            std::string args_str = match[2];
            std::string return_type = mapType(match[3]);
            std::string args;
            if (!args_str.empty()) {
                std::istringstream argStream(args_str);
                std::string arg;
                std::vector<std::string> argList;
                while (std::getline(argStream, arg, ',')) {
                    auto pos = arg.find(':');
                    std::string argName = trim(arg.substr(0, pos));
                    std::string argType = trim(arg.substr(pos + 1));
                    argList.push_back(mapType(argType) + " " + argName);
                }
                args = join(argList, ", ");
            }
            return {return_type + " " + name + "(" + args + ") {", true, false};
        }

        // Return
        if (line.rfind("return ", 0) == 0) {
            return {"return " + line.substr(7) + ";", false, true};
        }

        // Function call or raw expression
        return {line + ";", false, false};
    }

    std::string wrapWithMain(const std::string& code) {
        return "#include <iostream>\n#include <string>\nusing namespace std;\n\n" + code + "\nint main() {\n    cout << greet(\"Lustra\") << endl;\n    return 0;\n}\n";
    }

    std::string trim(const std::string& str) {
        const char* whitespace = " \t\n\r";
        size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, end - start + 1);
    }

    std::string join(const std::vector<std::string>& vec, const std::string& sep) {
        std::ostringstream oss;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) oss << sep;
            oss << vec[i];
        }
        return oss.str();
    }
};

// Example usage
int main() {
    std::string source = R"(
let name: String = \"Lustra\"
let age = 5
func greet(who: String): String =>
    return \"Hello, \" + who
)";

    LustraCompiler compiler;
    std::string output = compiler.compile(source);
    std::cout << "=== GENERATED C++ CODE ===\n" << output << std::endl;
    return 0;
}

std::tuple<std::string, bool, bool> compileLine(const std::string& line, bool inFunction) {
    std::smatch match;

    // Variable with type
    if (std::regex_match(line, match, std::regex(R"(let (\w+): (\w+) = (.+))"))) {
        std::string name = match[1];
        std::string ltype = match[2];
        std::string value = match[3];
        return {mapType(ltype) + " " + name + " = " + value + ";", false, false};
    }

    // Variable inferred
    if (std::regex_match(line, match, std::regex(R"(let (\w+) = (.+))"))) {
        std::string name = match[1];
        std::string value = match[2];
        return {"auto " + name + " = " + value + ";", false, false};
    }

    // Print
    if (std::regex_match(line, match, std::regex(R"(print\((.+)\))"))) {
        return {"std::cout << " + match[1].str() + " << std::endl;", false, false};
    }

    // If statement
    if (std::regex_match(line, match, std::regex(R"(if (.+) =>)"))) {
        std::string condition = match[1];
        return {"if (" + condition + ") {", true, false};
    }

    // Else statement
    if (std::regex_match(line, match, std::regex(R"(else =>)"))) {
        return {"else {", true, false};
    }

    // Return
    if (line.rfind("return ", 0) == 0) {
        return {"return " + line.substr(7) + ";", false, true};
    }

    // End block manually
    if (line == "end") {
        return {"}", false, true};
    }

    // Function definition
    if (std::regex_match(line, match, std::regex(R"(func (\w+)\((.*?)\): (\w+) =>)"))) {
        std::string name = match[1];
        std::string args_str = match[2];
        std::string return_type = mapType(match[3]);
        std::string args;
        if (!args_str.empty()) {
            std::istringstream argStream(args_str);
            std::string arg;
            std::vector<std::string> argList;
            while (std::getline(argStream, arg, ',')) {
                auto pos = arg.find(':');
                std::string argName = trim(arg.substr(0, pos));
                std::string argType = trim(arg.substr(pos + 1));
                argList.push_back(mapType(argType) + " " + argName);
            }
            args = join(argList, ", ");
        }
        return {return_type + " " + name + "(" + args + ") {", true, false};
    }

    // Fallback: expression or function call
    return {line + ";", false, false};
}

// Lustra Compiler (Prototype) - Written in C++
// Parses Lustra syntax and generates executable C++ code

#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

class LustraCompiler {
public:
    std::string compile(const std::string& source) {
        std::istringstream iss(source);
        std::string line;
        std::string compiled;
        int indent = 0;
        bool inBlock = false;

        while (std::getline(iss, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;

            auto [compiledLine, isStart, isEnd] = compileLine(trimmed, inBlock);

            if (isEnd && indent > 0) indent--;

            compiled += std::string(indent * 4, ' ') + compiledLine + "\n";

            if (isStart) {
                indent++;
                inBlock = true;
            }
            if (isEnd) inBlock = false;
        }

        return wrapWithMain(compiled);
    }

private:
    std::string mapType(const std::string& ltype) {
        if (ltype == "String") return "std::string";
        if (ltype == "Int") return "int";
        if (ltype == "Float") return "float";
        if (ltype == "Bool") return "bool";
        return "auto";
    }

    std::tuple<std::string, bool, bool> compileLine(const std::string& line, bool inBlock) {
        std::smatch match;

        if (std::regex_match(line, match, std::regex(R"(let (\w+): (\w+) = (.+))"))) {
            return {mapType(match[2]) + " " + match[1].str() + " = " + match[3].str() + ";", false, false};
        }

        if (std::regex_match(line, match, std::regex(R"(let (\w+) = (.+))"))) {
            return {"auto " + match[1].str() + " = " + match[2].str() + ";", false, false};
        }

        if (std::regex_match(line, match, std::regex(R"(print\((.+)\))"))) {
            return {"std::cout << " + match[1].str() + " << std::endl;", false, false};
        }

        if (std::regex_match(line, match, std::regex(R"(if (.+) =>)"))) {
            return {"if (" + match[1].str() + ") {", true, false};
        }

        if (std::regex_match(line, match, std::regex(R"(else =>)"))) {
            return {"else {", true, false};
        }

        if (line.rfind("return ", 0) == 0) {
            return {"return " + line.substr(7) + ";", false, false};
        }

        if (line == "end") {
            return {"}", false, true};
        }

        if (std::regex_match(line, match, std::regex(R"(func (\w+)\((.*?)\): (\w+) =>)"))) {
            std::string args;
            std::vector<std::string> argList;
            std::istringstream argStream(match[2]);
            std::string arg;
            while (std::getline(argStream, arg, ',')) {
                auto pos = arg.find(':');
                std::string argName = trim(arg.substr(0, pos));
                std::string argType = trim(arg.substr(pos + 1));
                argList.push_back(mapType(argType) + " " + argName);
            }
            args = join(argList, ", ");
            return {mapType(match[3]) + " " + match[1].str() + "(" + args + ") {", true, false};
        }

        return {line + ";", false, false};
    }

    std::string wrapWithMain(const std::string& code) {
        return "#include <iostream>\n#include <string>\nusing namespace std;\n\n" + code + "\nint main() {\n    cout << greet(\"Lustra\") << endl;\n    return 0;\n}\n";
    }

    std::string trim(const std::string& str) {
        const char* whitespace = " \t\n\r";
        size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, end - start + 1);
    }

    std::string join(const std::vector<std::string>& vec, const std::string& sep) {
        std::ostringstream oss;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) oss << sep;
            oss << vec[i];
        }
        return oss.str();
    }
};

// Example usage
int main() {
    std::string source = R"(
let name: String = \"Lustra\"
let age = 5
func greet(who: String): String =>
    return \"Hello, \" + who
end
if age > 3 =>
    print(\"Old enough\")
else =>
    print(\"Too young\")
end
)";

    LustraCompiler compiler;
    std::string output = compiler.compile(source);
    std::cout << "=== GENERATED C++ CODE ===\n" << output << std::endl;
    return 0;
}
