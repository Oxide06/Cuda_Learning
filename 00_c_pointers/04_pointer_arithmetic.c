#include<stdio.h>
#include<stdlib.h>

int main(){
  int arr[] = {1,2,3,4,5};
  int *ptr = arr;
  for(int i = 0; i < 5; i++){
    printf("Value and Address: %d %p\n", *ptr, ptr);
    ptr++;
  }
  return 0;
}
