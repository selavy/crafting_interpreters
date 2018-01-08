class DevonshireCream {
    serveOn() {
        return "Scones";
    }
}

print DevonshireCream; // Expect: "DevonshireCream"

class Bagel {}
var bagel = Bagel();
print bagel; // Expect: "Bagel instance"

class C {}
var c = C();
c.field = "Hello, World!";
print c.field;
