#include <iostream>
#include <bitset>
#include <limits>

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
    long num;
    std::cout << "Enter a number: ";
    std::cin >> num;
    binaryDouble(num);
  }
}