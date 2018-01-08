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

class Bacon { eat() { print "Crunch crunch crunch!"; } }
Bacon().eat(); // Prints "Crunch crunch crunch!".

class Cake {
    taste() {
        var adj = "declicous";
        print "The " + this.flavor + " cake is " + adj + "!";
    }
}

var cake = Cake();
cake.flavor = "German chocolate";
cake.taste(); // Prints "The German chocolate cake is declicious!".

class Thing {
    getCallback() {
        fun localFunction() {
            print this;
        }
        return localFunction;
    }
}

var callback = Thing().getCallback();
callback();

class Foo {
    init() {
        print this;
    }
}

var foo = Foo();

class Base {}
class Derived < Base {}
var derived = Derived();

class Doughnut {
    cook() {
        print "Fry until golden brown.";
    }
}
class BostonCream < Doughnut {}
BostonCream().cook();
