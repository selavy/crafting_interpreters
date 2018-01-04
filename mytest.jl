var a = "global a";
var b = "global b";
var c = "global c";
{
    var a = "outer a";
    var b = "outer b";
    {
        var a = "inner a";
        print a;
        print b;
        print c;
    }
    print a;
    print b;
    print c;
}
print a;
print b;
print c;
if(true) {
    print "hello";
    print "This is inside a block!";
}
if(false)
    print "Branch 1 executed";
else
    print "Branch 2 executed";

print "hi" or 2; // "hi".
print nil or "yes"; // "yes".

var i = 1;
while (i < 5) {
    print i;
    i = i + 1;
}

print "Beginning for loop...";
for (var i = 0; i < 10; i = i + 1) print i;
