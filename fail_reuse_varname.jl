fun bad() {
    var a = "first";
    var a = "second"; // <-- boom!
    // Expect: "Variable with this name already declared in this scope."
}
bad();
