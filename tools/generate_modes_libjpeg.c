/*
 * Writes the fixtures of tests/extended_modes.rs: synthetic pictures coded by
 * libjpeg-turbo (3.1.0 was used) in the modes `cjpeg` has no switches for, or
 * not in this combination. With `raw` it prints the FNV-1a hash of what
 * libjpeg-turbo decodes a file to (samples as little-endian u16, in the
 * library's own CMYK for four components), which is what the tests expect.
 *
 *   cmake -S libjpeg-turbo-3.1.0 -B b -DENABLE_SHARED=0 && make -C b jpeg-static
 *   cc -O2 -o modes generate_modes_libjpeg.c -Ilibjpeg-turbo-3.1.0/src -Ib b/libjpeg.a -lm
 *   ./modes gen <directory>
 *   ./modes raw <file>...
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jpeglib.h"

typedef struct {
  const char *name;
  int w, h, components, precision, hsamp, vsamp, progressive, arithmetic;
  int restart, separate_scans, dac, predictor, point_transform, ycck;
} Picture;

static const Picture pictures[] = {
  /*                          w   h  n  P  H  V prog ari rst sep dac psv pt ycck */
  { "seq8_420_huffman",      35, 23, 3,  8, 2, 2, 0, 0, 3, 0, 0, 0, 0, 0 },
  { "seq8_420_arithmetic",   35, 23, 3,  8, 2, 2, 0, 1, 3, 0, 1, 0, 0, 0 },
  { "prog8_422_huffman",     35, 23, 3,  8, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
  { "prog8_422_arithmetic",  35, 23, 3,  8, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
  { "seq12_420_huffman",     35, 23, 3, 12, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0 },
  { "seq12_440_separate",    35, 23, 3, 12, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0 },
  { "prog12_422_huffman",    35, 23, 3, 12, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
  { "seq12_grey_arithmetic", 35, 23, 1, 12, 1, 1, 0, 1, 3, 0, 1, 0, 0, 0 },
  { "prog12_420_arithmetic", 35, 23, 3, 12, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0 },
  { "seq12_ycck",            19, 11, 4, 12, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1 },
  { "lossless14_pair",       24, 16, 2, 14, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0 },
  { "lossless12_pair_scans", 24, 16, 2, 12, 1, 1, 0, 0, 1, 1, 0, 6, 1, 0 },
};

static unsigned random_state;

static int sample(const Picture *p, int x, int y, int c)
{
  int max = (1 << p->precision) - 1;
  double v = 0.5 + 0.3 * sin(x * (0.11 + 0.05 * c) + c) * cos(y * 0.13) +
             0.15 * (((x / 7 + y / 5 + c) & 1) ? 1 : -1);
  random_state = random_state * 1103515245u + 12345u;
  v += ((int)((random_state >> 16) % 2001) - 1000) / 12000.0;
  int s = (int)(v * max + 0.5);
  if ((x * 31 + y * 17) % 97 == 0) s = max;
  if ((x * 13 + y * 37) % 101 == 0) s = 0;
  return s < 0 ? 0 : s > max ? max : s;
}

static void generate(const char *directory, const Picture *p)
{
  struct jpeg_compress_struct c;
  struct jpeg_error_mgr e;
  jpeg_scan_info scans[4];
  char path[1024];
  snprintf(path, sizeof path, "%s/%s.jpg", directory, p->name);
  FILE *f = fopen(path, "wb");

  random_state = 12345;
  c.err = jpeg_std_error(&e);
  jpeg_create_compress(&c);
  jpeg_stdio_dest(&c, f);
  c.image_width = p->w;
  c.image_height = p->h;
  c.input_components = p->components;
  c.in_color_space = p->components == 1 ? JCS_GRAYSCALE :
                     p->components == 2 ? JCS_UNKNOWN :
                     p->components == 3 ? JCS_RGB : JCS_CMYK;
  c.data_precision = p->precision;
  jpeg_set_defaults(&c);
  c.data_precision = p->precision;
  if (p->predictor) {
    jpeg_enable_lossless(&c, p->predictor, p->point_transform);
  } else {
    jpeg_set_quality(&c, 90, TRUE);
    if (p->ycck) jpeg_set_colorspace(&c, JCS_YCCK);
    for (int i = 0; i < p->components; i++) {
      int full = i == 0 || i == 3;
      c.comp_info[i].h_samp_factor = full ? p->hsamp : 1;
      c.comp_info[i].v_samp_factor = full ? p->vsamp : 1;
    }
  }
  c.arith_code = p->arithmetic;
  c.optimize_coding = !p->arithmetic;
  if (p->dac)
    for (int i = 0; i < NUM_ARITH_TBLS; i++) {
      c.arith_dc_L[i] = 1;
      c.arith_dc_U[i] = 4;
      c.arith_ac_K[i] = 12;
    }
  if (p->progressive) jpeg_simple_progression(&c);
  if (p->separate_scans) {
    memset(scans, 0, sizeof scans);
    for (int i = 0; i < p->components; i++) {
      scans[i].comps_in_scan = 1;
      scans[i].component_index[0] = i;
      scans[i].Ss = p->predictor;
      scans[i].Se = p->predictor ? 0 : 63;
      scans[i].Al = p->point_transform;
    }
    c.scan_info = scans;
    c.num_scans = p->components;
  }
  if (p->predictor) c.restart_in_rows = p->restart;
  else c.restart_interval = p->restart;

  jpeg_start_compress(&c, TRUE);
  for (int y = 0; y < p->h; y++) {
    JSAMPLE row8[256];
    J12SAMPLE row12[256];
    J16SAMPLE row16[256];
    for (int i = 0; i < p->w * p->components; i++) {
      int s = sample(p, i / p->components, y, i % p->components);
      row8[i] = s;
      row12[i] = s;
      row16[i] = s;
    }
    JSAMPROW r8 = row8;
    J12SAMPROW r12 = row12;
    J16SAMPROW r16 = row16;
    if (p->precision <= 8) jpeg_write_scanlines(&c, &r8, 1);
    else if (p->precision <= 12) jpeg12_write_scanlines(&c, &r12, 1);
    else jpeg16_write_scanlines(&c, &r16, 1);
  }
  jpeg_finish_compress(&c);
  jpeg_destroy_compress(&c);
  fclose(f);
}

static void print_hash(const char *path)
{
  struct jpeg_decompress_struct c;
  struct jpeg_error_mgr e;
  FILE *f = fopen(path, "rb");
  unsigned long long hash = 0xcbf29ce484222325ull;

  c.err = jpeg_std_error(&e);
  jpeg_create_decompress(&c);
  jpeg_stdio_src(&c, f);
  jpeg_read_header(&c, TRUE);
  if (c.num_components == 4) c.out_color_space = JCS_CMYK;
  jpeg_start_decompress(&c);
  while (c.output_scanline < c.output_height) {
    JSAMPLE row8[1024];
    J12SAMPLE row12[1024];
    J16SAMPLE row16[1024];
    JSAMPROW r8 = row8;
    J12SAMPROW r12 = row12;
    J16SAMPROW r16 = row16;
    if (c.data_precision <= 8) jpeg_read_scanlines(&c, &r8, 1);
    else if (c.data_precision <= 12) jpeg12_read_scanlines(&c, &r12, 1);
    else jpeg16_read_scanlines(&c, &r16, 1);
    for (unsigned i = 0; i < c.output_width * c.output_components; i++) {
      unsigned s = c.data_precision <= 8 ? row8[i] :
                   c.data_precision <= 12 ? (unsigned)row12[i] : row16[i];
      hash = (hash ^ (s & 0xff)) * 0x100000001b3ull;
      hash = (hash ^ (s >> 8)) * 0x100000001b3ull;
    }
  }
  printf("%s %dx%dx%d %016llx\n", path, c.output_width, c.output_height,
         c.output_components, hash);
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  fclose(f);
}

int main(int argc, char **argv)
{
  if (argc == 3 && !strcmp(argv[1], "gen")) {
    for (size_t i = 0; i < sizeof pictures / sizeof *pictures; i++)
      generate(argv[2], &pictures[i]);
  } else if (argc > 2 && !strcmp(argv[1], "raw")) {
    for (int i = 2; i < argc; i++) print_hash(argv[i]);
  } else {
    return 1;
  }
  return 0;
}
