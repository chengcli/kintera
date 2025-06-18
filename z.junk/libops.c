#include <stdio.h>

void f1(int x) {
    printf("f1 called with %d\n", x);
}

void f2(int x) {
    printf("f2 called with %d\n", x);
}

void call_with_callback(void (*func)(int)) {
    for (int i = 0; i < 3; ++i) {
        func(i);
    }
}
