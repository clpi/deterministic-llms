// llama.cpp driver for the value test's 14B row (tasks/VALUE_AMEND_2.json).
// Derived from harness/xhw_llama.cpp, plus: a context size argument, saving and
// restoring sequence 0's KV state (so every path starts from the same prompt
// state), and detokenization.
//
//   value_llama MODEL.gguf THREADS NCTX
//
// stdin commands, one per line; replies on stdout:
//   V                 -> "<n_vocab>\n"
//   K <nbytes>\n<raw> -> "<n> id1 id2 ...\n"   (tokenize with the model's special tokens)
//   R                 -> "ok\n"                  (clear the KV memory)
//   D <n> id1 ... idn -> n_vocab float32 values, little-endian (logits after the last token)
//   S                 -> "ok\n"                  (save sequence 0's state and position)
//   L                 -> "ok\n"                  (restore the saved state and position)
//   T <n> id1 ... idn -> "<nbytes>\n<raw utf-8>" (detokenize, special tokens rendered)
#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL THREADS NCTX\n", argv[0]); return 2; }
    const int threads = atoi(argv[2]);
    const int nctx = atoi(argv[3]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx;
    cp.n_batch = nctx;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    llama_pos pos = 0;
    std::vector<uint8_t> saved;
    llama_pos saved_pos = 0;
    static char line[1 << 20];
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'V') {
            printf("%d\n", n_vocab);
        } else if (line[0] == 'R') {
            llama_memory_clear(llama_get_memory(ctx), true);
            pos = 0;
            printf("ok\n");
        } else if (line[0] == 'S') {
            saved.resize(llama_state_seq_get_size(ctx, 0));
            const size_t got = llama_state_seq_get_data(ctx, saved.data(), saved.size(), 0);
            saved.resize(got);
            saved_pos = pos;
            printf("ok\n");
        } else if (line[0] == 'L') {
            llama_memory_clear(llama_get_memory(ctx), true);
            if (llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0) == 0) { fprintf(stderr, "restore failed\n"); return 1; }
            pos = saved_pos;
            printf("ok\n");
        } else if (line[0] == 'T') {
            char * p = line + 2;
            const int n = (int) strtol(p, &p, 10);
            std::string out;
            char buf[256];
            for (int i = 0; i < n; ++i) {
                const llama_token t = (llama_token) strtol(p, &p, 10);
                const int k = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
                if (k > 0) out.append(buf, k);
            }
            printf("%zu\n", out.size());
            fwrite(out.data(), 1, out.size(), stdout);
        } else if (line[0] == 'K') {
            const int nb = atoi(line + 2);
            std::string text(nb, '\0');
            if (nb && fread(&text[0], 1, nb, stdin) != (size_t) nb) return 1;
            const int n = -llama_tokenize(vocab, text.data(), nb, nullptr, 0, true, false);
            std::vector<llama_token> toks(n);
            if (llama_tokenize(vocab, text.data(), nb, toks.data(), n, true, false) < 0) return 1;
            printf("%d", n);
            for (auto t : toks) printf(" %d", t);
            printf("\n");
        } else if (line[0] == 'D') {
            char * p = line + 2;
            const int n = (int) strtol(p, &p, 10);
            llama_batch_ext_clear(batch);
            for (int i = 0; i < n; ++i) {
                const llama_token t = (llama_token) strtol(p, &p, 10);
                const int32_t idx = llama_batch_ext_add_token(batch, 0, t);
                const llama_pos ps = pos + i;
                llama_batch_ext_set_pos(batch, idx, &ps);
            }
            llama_batch_ext_set_output_logits(batch, n - 1, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
            pos += n;
            const float * lg = llama_get_logits_ith(ctx, -1);
            fwrite(lg, sizeof(float), n_vocab, stdout);
        } else {
            continue;
        }
        fflush(stdout);
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
