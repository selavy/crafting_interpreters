fun mkcounter() {
    var i = 0;
    fun count() {
        i = i + 1;
        print i;
    }
    return count;
}

var counter = mkcounter();
counter();
counter();
