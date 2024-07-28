import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math';

import 'package:dart_llama_lib/bind/llama_cpp.dart';
import 'package:ffi/ffi.dart';

/// An enumeration representing different types of Llama Rope Scaling.
enum RopeScalingType { unspecified, none, linear, yarn, maxValue }

class Llama{
  static llama_cpp? _lib;
  String loraBase;
  List<(String, double)> loraAdapters;

  /// Pointer to Llama model
  late Pointer<llama_model> model;

  /// Pointer to Llama context
  late Pointer<llama_context> context;

  /// Batch configuration to Llama model
  late llama_batch batch;

  /// Length of the ouput. Default is -1.
  int length = -1;

  /// Cursor position in the token list. Default is 0.
  int tokenPos = 0;
  List<int> lastTokens = [];

  /// [EXPERIMENTAL]
  /// -------------------------------------------------------------------------
  final StreamController<String> _controller = StreamController.broadcast();

  Stream<String> get stream => _controller.stream;

  ///--------------------------------------------------------------------------


  /// Getter for the Llama library path.
  static String? libPath;

  /// Getter for the Llama library.
  /// Loads the library based on the current platform
  llama_cpp get lib{
    if(Platform.isAndroid){
      _lib = llama_cpp(DynamicLibrary.open('libllama.so'));
    }
    else if(libPath != null && File(libPath!).existsSync()){
      _lib = llama_cpp(DynamicLibrary.open(libPath!));
    }
    else{
        _lib = llama_cpp(DynamicLibrary.process());
    }

    return _lib!;
  }

  /// Llama constructor:
  /// Initialize a instance with the Llama model and parameters.
  Llama({
    required String modelPath,
    int mainGpu = 0,
    int gpuLayer = 99,
    bool mLock = false,
    bool mMap = true,
    bool vOnly = false,

    int seed = -1,
    int nCtx = 512,
    int nBatch = 512,
    int nThreads = 4,
    double ropeFreqBase = 0.0,
    double ropeFreqScale = 0.0,
    double yarnExtFactor = -1.0,
    double yarnAttnFactor = 1.0,
    double yarnBetaFast = 32.0,
    double yarnBetaSlow = 1.0,
    int yarnOrigCtx = 0,
    bool logitsAll = false,
    bool embedding = false,
    bool offloadKqv = true,

    String grammar = '', // toNativeUtf8()
    String cfgNegativePrompt = '', // toNativeUtf8()
    double cfgScale = 1.0,

    this.loraBase = '',
    this.loraAdapters = const []
  }){
    lib.llama_backend_init();
    llama_model_params modelParams = setModelParams(mainGpu, gpuLayer, mLock, mMap, vOnly);
    Pointer<Char> char = modelPath.toNativeUtf8().cast<Char>();
    model = lib.llama_load_model_from_file(char, modelParams);
    malloc.free(char);
    if(model.address == 0){
      throw Exception('Failed to load model at $modelPath');
    }
    
    llama_context_params ctxParams = setContextParams(seed,
      nCtx,
      nBatch,
      nThreads,
      ropeFreqBase,
      ropeFreqScale,
      yarnExtFactor,
      yarnAttnFactor,
      yarnBetaFast,
      yarnBetaSlow,
      yarnOrigCtx,
      logitsAll,
      embedding,
      offloadKqv
    );
    context = lib.llama_new_context_with_model(model, ctxParams);
    if(context.address == 0){
      throw Exception('Failed to load context');
    }

    batch = lib.llama_batch_init(nBatch, 0, 1);

    Pointer<Char> lora = loraBase.toNativeUtf8().cast<Char>();
    for(int i=0; i < loraAdapters.length; i++){
      Pointer<Char> loraAdapter = loraAdapters[i].$1.toNativeUtf8().cast<Char>();
      double loraScale = loraAdapters[i].$2;
      int _ = lib.llama_model_apply_lora_from_file(model,
        lora,
        loraScale,
        loraBase.isNotEmpty ? lora : nullptr,
        nThreads
      );
      malloc.free(loraAdapter);
      if(_ != 0){
        lib.llama_batch_free(batch);
        lib.llama_free(context);
        lib.llama_free_model(model);
        throw Exception('Failed to load lora adapter.');
      }
    }
    malloc.free(lora);
  }

  /// Cleans the Llama instance and relase all allocated resorces.
  void dispose(){
    lib.llama_batch_free(batch);

    if(context.address != 0){
      lib.llama_free(context);
    }

    if(model.address != 0){
      lib.llama_free_model(model);
    }

    lib.llama_backend_free();
  }

  /// Generates a stream of text based on a given prompt.
  /// It continues generating text until an end-of-sequence condition is met.
  String prompt({
    required String text,
    bool penalizeNl = true,
    int nPrev = 64,
    int nProbs = 0,
    int topK = 40,
    double topP = 0.95,
    double minP = 0.05,
    double tfsZ = 1.0,
    double typicalP = 1.0,
    double temp = 0.8,
    int penaltyLastN = 64,
    double penaltyRepeat = 1.1,
    double penaltyFreq = 0.0,
    double penaltyPresent = 0.0,
    int mirostat = 0,
    double mirostatTau = 5.0,
    double mirostatEta = 0.1
  }) //async*{
  {
    List<String> output = [];
    lastTokens = tokenize(text, true);
    if(length != -1){
      int nCtx = lib.llama_n_ctx(context);
      int nKv = lastTokens.length + (length - lastTokens.length);

      if(nKv > nCtx){
        throw Exception('Error: Not enough KV cache');
      }
    }

    batch.n_tokens = 0;

    for(int i=0; i < lastTokens.length; i++){
      addBatch(batch, lastTokens[i], i, [0], false);
    }

    batch.logits[batch.n_tokens - 1] = 1;

    if(lib.llama_decode(context, batch) != 0){
      throw Exception('Error: llama_decode() failed');
    }

    tokenPos = batch.n_tokens;
    while(true){
      var (result, isDone) = getGenerated(nPrev, penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent, topP, topK, temp);
      output.add(result);
      //_controller.add(result);

      if(isDone){
        break;
      }
    }

    return output.join('');
  }


  /// [EXPERIMENTAL]
  /// -------------------------------------------------------------------------
  Stream<void> promptAsync({
    required String text,
    bool penalizeNl = true,
    int nPrev = 64,
    int nProbs = 0,
    int topK = 40,
    double topP = 0.95,
    double minP = 0.05,
    double tfsZ = 1.0,
    double typicalP = 1.0,
    double temp = 0.8,
    int penaltyLastN = 64,
    double penaltyRepeat = 1.1,
    double penaltyFreq = 0.0,
    double penaltyPresent = 0.0,
    int mirostat = 0,
    double mirostatTau = 5.0,
    double mirostatEta = 0.1
  })
  async*{
    lastTokens = tokenize(text, true);
    if(length != -1){
      int nCtx = lib.llama_n_ctx(context);
      int nKv = lastTokens.length + (length - lastTokens.length);

      if(nKv > nCtx){
        throw Exception('Error: Not enough KV cache');
      }
    }

    batch.n_tokens = 0;

    for(int i=0; i < lastTokens.length; i++){
      addBatch(batch, lastTokens[i], i, [0], false);
    }

    batch.logits[batch.n_tokens - 1] = 1;

    if(lib.llama_decode(context, batch) != 0){
      throw Exception('Error: llama_decode() failed');
    }

    tokenPos = batch.n_tokens;
    while(true){
      var (result, isDone) = getGenerated(nPrev, penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent, topP, topK, temp);
      _controller.add(result);

      if(isDone){
        break;
      }
    }
  }
  /// -------------------------------------------------------------------------


  /// Reset the state of the Llama instance
  void clear(){
    lastTokens.clear();
    lib.llama_kv_cache_clear(context);
    batch.n_tokens = 0;
    tokenPos = 0;
  }

  /// Internal method:
  /// Set context parameters to llama instance
  llama_context_params setContextParams(
    int seed,
    int nCtx,
    int nBatch,
    int nThreads,
    double ropeFreqBase,
    double ropeFreqScale,
    double yarnExtFactor,
    double yarnAttnFactor,
    double yarnBetaFast,
    double yarnBetaSlow,
    int yarnOrigCtx,
    bool logitsAll,
    bool embedding,
    bool offloadKqv,
  ){
    llama_context_params ctx = lib.llama_context_default_params();
    ctx.seed = (seed == -1) ? Random().nextInt(1000000) : seed;   // Seed for random number generation. (-1 = Random number)
    ctx.n_ctx = nCtx;                                             // Text context size. (Default is 512, 0 uses model's default)
    ctx.n_batch = nBatch;                                         // Maximum batch size for prompt processing. (Default is 512)
    ctx.n_threads = nThreads;                                     // Number of threads to use for generation. (Default is 4)
    ctx.n_threads_batch = nThreads;                               // Number of threads to use for batch processing. (Default is 4)
    ctx.rope_freq_base = ropeFreqBase;                            // Base frequency for RoPE (Rotary Positional Embedding). (0.0 = model's default)
    ctx.rope_freq_scale = ropeFreqScale;                          // Frequency scaling factor for RoPE. (0.0 = model's default)
    ctx.yarn_ext_factor = yarnExtFactor;                          // YaRN extrapolation mix factor. (-1.0 = model's default)
    ctx.yarn_attn_factor = yarnAttnFactor;                        // YaRN attention magnitude scaling factor. (Default is 1.0)
    ctx.yarn_beta_fast = yarnBetaFast;                            // YaRN low correction dimension. (Default is 32.0)
    ctx.yarn_beta_slow = yarnBetaSlow;                            // YaRN high correction dimension. (Default is 1.0)
    ctx.yarn_orig_ctx = yarnOrigCtx;                              // Original context size for YaRN. (Default is 0)
    ctx.logits_all = logitsAll;                                   // Computes all logits, not just the last one. (Default is false)
    ctx.embeddings = embedding;                                   // Operates in embedding mode only. (Default is falsse)
    ctx.offload_kqv = offloadKqv;                                 // Determines whether to offload the KQV operations to GPU. (Default is true)
    ctx.rope_scaling_type = RopeScalingType.unspecified.index;    // Type of RoPE scaling to use (Default is 'unspecified')
    
    return ctx;
  }

  /// Internal method:
  /// Set model parameters to llama instance
  llama_model_params setModelParams(
    int mainGpu,
    int gpuLayer,
    bool mLocK,
    bool mMap,
    bool vOnly,
  ){
    llama_model_params p = lib.llama_model_default_params();
    p.main_gpu = mainGpu;                                         // Use chosen GPU
    p.n_gpu_layers = gpuLayer;                                    // Number of layers to store in VRAM. (Default is 99)
    p.use_mlock = mLocK;                                          // Force the system to keep the model in RAM. (Default is false)
    p.use_mmap = mMap;                                            // Use memory mapping if possible. (Default is true)
    p.vocab_only = vOnly;                                         // Only load the vocabulary without weights. (Defaut is false)

    return p;
  }

  /// Internal method:
  /// Converts a text string to a list of token IDs.
  /// An optional flag 'setBos' indicates whether to prepend a beginning-of-sentence token.
  /// Returns a list of integers representing tokens.
  List<int> tokenize(String text, bool setBos){
    Pointer<Char> cChar = text.toNativeUtf8().cast<Char>();
    int textLen = utf8.encode(text).length;
    int tokensLen = textLen + (setBos ? 1 : 0) + 1;
    Pointer<llama_token> tokens = malloc.allocate<llama_token>(tokensLen * sizeOf<llama_token>());

    try{
      int tokCount = lib.llama_tokenize(model, cChar, textLen, tokens, tokensLen, setBos, false);

      List<int>tokenList = [];
      for(int i=0; i < tokCount; i++){
        tokenList.add(tokens[i]);
      }

      return tokenList;
    }
    finally{
      malloc.free(cChar);
      malloc.free(tokens);
    }
  }

  /// Internal method:
  /// Adds a token to the batch for processing.
  /// Appends a token with its associated position and sequence IDs to the batch.
  /// The 'logits' flag indicates whether logits should be calculated for this token.
  void addBatch(
    llama_batch batch,
    int id,
    int pos,
    List<int> seqId,
    bool logits
  ){
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seqId.length;
    for(int i=0; i < seqId.length; i++){
      batch.seq_id[batch.n_tokens][i] = seqId[i];
    }
    batch.logits[batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
  }

  /// Internal method:
  /// Converts a token ID to its corresponding string representation.
  /// It takes a token ID and returns the associated text piece.
  /// It handles the conversion and memory management involved in this process.
  /// This is typically used in decoding the output of the model.
  String tokenToPiece(int token){
    int bufferSize = 64;
    Pointer<Char> result = malloc.allocate<Char>(bufferSize);
    try{
      int pieces = lib.llama_token_to_piece(model, token, result, bufferSize, 0, false);
      pieces = min(pieces, bufferSize - 1);

      final buffer = result.cast<Uint8>().asTypedList(pieces);

      return utf8.decode(buffer, allowMalformed: true);
    }
    finally{
      malloc.free(result);
    }
  }

  /// Internal method:
  /// Generates and returns the next token in the sequence based on the current context.
  /// This function handles the selection and decoding of the next token.
  /// Returns a tuple with the generated text and a boolean indicating if the end-of-sequence token is reached.
  /// An exception is thrown if llama_decode fails during processing.
  (String, bool) getGenerated(
    int nPrev,
    int penaltyLastN,
    double penaltyRepeat,
    double penaltyFreq,
    double penaltyPresent,
    double topP,
    int topK,
    double temp
  ){
    Pointer<Int32> newTokenId = calloc<Int32>();
    
    final int nVocab = lib.llama_n_vocab(model);
    
    final logits = lib.llama_get_logits(context);
    
    final Pointer<llama_token_data> candidates = calloc<llama_token_data>(nVocab);
    for(int tokenId=0; tokenId < nVocab; tokenId++){
      candidates[tokenId].id = tokenId;
      candidates[tokenId].logit = logits[tokenId];
      candidates[tokenId].p = 0.0;
    }

    final Pointer<llama_token_data_array>candidatesP = calloc<llama_token_data_array>();
    candidatesP.ref.data = candidates;
    candidatesP.ref.size = nVocab;
    candidatesP.ref.sorted = false;

    int minSize = min(nPrev, lastTokens.length);
    Pointer<Int32> lastTokensP = calloc<Int32>(nPrev);
    List<int> safeLastTokens = lastTokens.take(minSize).toList();
    lastTokensP.asTypedList(minSize).setAll(0, safeLastTokens);
    lib.llama_sample_repetition_penalties(
      context,
      candidatesP,
      lastTokensP,
      penaltyLastN,
      penaltyRepeat,
      penaltyFreq,
      penaltyPresent
    );
    lib.llama_sample_top_k(context, candidatesP, topK, 1);
    lib.llama_sample_top_p(context, candidatesP, topP, 1);
    lib.llama_sample_temp(context, candidatesP, temp);

    newTokenId.value = lib.llama_sample_token(context, candidatesP);
    
    bool isEOSToken = newTokenId.value == lib.llama_token_eos(model);
    String newTokenStr = '';

    if(newTokenId.value != lib.llama_token_bos(model)){
      newTokenStr = tokenToPiece(newTokenId.value);
    }

    batch.n_tokens = 0;
    addBatch(batch, newTokenId.value, tokenPos, [0], true);
    
    lastTokens.add(newTokenId.value);

    tokenPos++;

    if(lib.llama_decode(context, batch) != 0){
      throw Exception('Error: llama_decode() failed.');
    }

    calloc.free(newTokenId);
    calloc.free(candidates);
    calloc.free(candidatesP);
    calloc.free(lastTokensP);

    return (newTokenStr, isEOSToken);
  }
}