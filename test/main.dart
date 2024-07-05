import '../lib/src/llama.dart';

void main() {
  print('starting model...');

  Llama model = Llama('f:\\meus arquivos\\IA\\stablelm-2-zephyr-1_6b-Q4_0.gguf');

  print('seting test prompt: "hello there!"');

  print(model.prompt('hello there!'));

  print('done');
}