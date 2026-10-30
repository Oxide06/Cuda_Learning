#include<stdio.h>
#include<stdlib.h>

int main(){
  int *ptr = NULL;
  printf("1. Intial value of ptr: %p\n", (void*)ptr);

  if(ptr == NULL){
    printf("2. ptr is NULL, cannot dereference\n");
  }
  ptr = malloc(sizeof(int));
  if(ptr == NULL){
    printf("3. Memory allocation failed\n");
    return 1;
  }
  *ptr = 10;
  printf("Value of ptr: %p\n", ptr);
}
