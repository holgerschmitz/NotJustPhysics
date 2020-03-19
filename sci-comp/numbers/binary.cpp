#include <iostream>
#include <bitset>

void binaryDouble(double num) {
  long *lnum = (long*)&num;
  std::cout << std::bitset<64>(*lnum) << std::endl;
}

void binaryLong(long num) {
  std::cout << std::bitset<64>(num) << std::endl;
}

int main() {
  while (std::cin) {
    double num;
    std::cout << "Enter a number: ";
    std::cin >> num;
    binaryDouble(num);
  }
}