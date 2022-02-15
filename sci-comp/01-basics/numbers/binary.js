const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function binaryDouble(num) {
  const buf=Buffer.alloc(8); // 8 bytes needed for double
  buf.writeDoubleBE(num);
  return Array.from(buf.values()).map(b => b.toString(2).padStart(8,'0')).join();
}

function binaryInt(num) {
  const buf=Buffer.alloc(8); // 8 bytes needed for double
  buf.writeBigInt64BE(num);
  return Array.from(buf.values()).map(b => b.toString(2).padStart(8,'0')).join();
}

function main_loop_double() {
  rl.question('Enter a number: ', (line) => {
    const num = parseFloat(line);
    console.log(binaryDouble(num));
    main_loop_double();
  });
};

function main_loop_int() {
  rl.question('Enter a number: ', (line) => {
    const num = BigInt(line);
    console.log(binaryInt(num));
    main_loop_int();
  });
};

main_loop_int();