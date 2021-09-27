#include <iostream>
#include <bitset>
#include <limits>
#include <cmath>

void binaryFloat(float num) {
  int32_t *inum = (int32_t*)&num;
  std::cout << std::bitset<32>(*inum) << std::endl;
}

void binaryDouble(double num) {
  long *lnum = (long*)&num;
  std::cout << std::bitset<64>(*lnum) << std::endl;
}

void binaryLong(long num) {
  std::cout << std::bitset<64>(num) << std::endl;
}

int main() {
  std::cout << "C++ integer types:" << std::endl;
  std::cout << "int: " << sizeof(int) << " " << std::numeric_limits<int>::min << " " << std::numeric_limits<int>::max << std::endl;
  while (std::cin) {
    float num;
    std::cout << "Enter a number: ";
    std::cin >> num;
    binaryFloat(num);
  }
}