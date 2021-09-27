#include <iostream>
#include <limits>


int main() {
  std::cout << "C++ integer types:" << std::endl;
  std::cout << "int: " << sizeof(int) << " " << std::numeric_limits<int>::min() << " " << std::numeric_limits<int>::max() << std::endl;
  std::cout << "long: " << sizeof(long) << " " << std::numeric_limits<long>::min() << " " << std::numeric_limits<long>::max() << std::endl;
  std::cout << "long long: " << sizeof(long long) << " " << std::numeric_limits<long long>::min() << " " << std::numeric_limits<long long>::max() << std::endl;
}