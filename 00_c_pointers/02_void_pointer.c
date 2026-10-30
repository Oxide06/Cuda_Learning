#include<stdio.h>

int main(){
  int num = 10;
  float fnum = 3.14;
  void *ptr;
  ptr = &num;
  printf("Integer: %d\n",*(int*)ptr);
  /*
  *We can't deference a void pointer because it doesn't know what type it is pointing to.
  *So, we need to cast it to the appropriate type before dereferencing it.
  *In this case, we are dereferencing an integer pointer --> (int*) and expecting an integer value.
  */
  ptr = &fnum;
  printf("Float: %f\n",*(float*)ptr);
  return 0;
}
