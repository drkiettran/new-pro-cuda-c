#include <stdio.h>

void say_hello(char* name) {
    printf("Hello, %s\n", name);
}

int main(int argc, char** argv) {
    if (argc > 1) {
        say_hello(argv[1]);
    } else {
        say_hello("friend");
    }
}