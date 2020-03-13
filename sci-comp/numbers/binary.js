const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function binary(num) {
    const buf=Buffer.alloc(8); // 8 bytes needed for double
    buf.writeDoubleBE(num);
    return Array.from(buf.values()).map(b => b.toString(2).padStart(8,'0')).join();
}

function main_loop() {
    rl.question('Enter a number: ', (line) => {
        const num = parseFloat(line);
        console.log(binary(num));
        main_loop();
    });
};

main_loop();