#include <stdio.h>

int main() {
    int x = 10;
    int *ptr = &x;
    int **ptr2 = &ptr;
    int ***ptr3 = &ptr2;
    printf("Value of x: %d\n", x);
    printf("Address of x: %p\n", ptr);
    printf("Value of ptr: %d\n", *ptr);
    printf("Address of ptr: %p\n", ptr2);
    printf("Value of ptr2: %d\n", **ptr2);
    printf("Address of ptr2: %p\n", ptr3);
    printf("Value of ptr3: %d\n", ***ptr3);
    return 0;
}
