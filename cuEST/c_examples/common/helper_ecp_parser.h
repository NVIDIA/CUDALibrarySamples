/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMON_HELPER_ECP_PARSER
#define COMMON_HELPER_ECP_PARSER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This helper provides a simple parser for ECP files. It reads through an
 * ECP file until a specified element is found, then returns the basis set
 * definition for that element in the ECPSet_t structure.
 *
 * The ECP format is the Gaussian94 basis set format. Files in this format
 * can be downloaded from the EMSL basis set exchange (in either "Gaussian"
 * or "Psi4" format). Note that this parser does not handle SP or SPD shells.
 * To avoid their use, specify the "Uncontract SPDF" option if downloading from
 * EMSL. Otherwise, split the SP definition into separate S and P shell definitions.
 *
 * Comment lines starting with "!" are ignored by the parser. ECP definitions may
 * be present at the end of the file (i.e. after the basis set definition) or may be
 *  in a standalone file.
 */

typedef struct {
    size_t   n_shells;           // Total number of shells
    size_t   n_elec;             // Number of electrons
    uint64_t *shell_types;       // e.g. {'S=0','P=1','D=2',...}
    uint64_t *num_primitives;    // primitives per shell
    uint64_t *primitive_offsets; // index in exponents/coeffs arrays
    uint64_t *Ns;                // all Ns, packed
    double   *exponents;         // all exponents, packed
    double   *coefficients;      // all coefficients, packed
} ECPShellSet_t;

/* Convert to uppercase (in-place) */
static void ecp_to_upper(char *s)
{
    while (*s) {
        *s = (char)toupper((unsigned char)*s);
        s++;
    }
}

/* Converts string representation of a float into a double */
static double ecp_parse_float(const char *str) {
    char buffer[128];
    size_t i;

    // Copy to a buffer we can modify */
    strncpy(buffer, str, sizeof(buffer));
    buffer[sizeof(buffer) - 1] = '\0';

    for (i = 0; buffer[i]; i++) {
        if (buffer[i] == 'd' || buffer[i] == 'D') {
            buffer[i] = 'e';
        }
    }

    return strtod(buffer, NULL);
}

static uint64_t ecp_count_words(const char *str) {
    uint64_t count = 0;
    uint64_t flag = 0;

    while (*str) {
        if (isspace((unsigned char) *str)) {
            flag = 0;
        } else {
            if (flag == 0) {
                count++;
                flag = 1;
            }
        }
        str++;
    }

    return count;
}

static uint64_t ecp_angular_momentum_type_to_L(char am) 
{
    const char shell_types_list[] = {
        'S','P','D','F','G','H','I','K',
        'L','M','N','O','Q','R','T','U','V','W','X','Y','Z'
    };
    am = toupper((unsigned char) am);
    for (uint64_t i = 0; i < 21; i++) {
        if (shell_types_list[i] == am) {
            return i;
        }
    }
    fprintf(stderr, "Unknown angular momentum symbol\n");
    exit(EXIT_FAILURE);
}

static ECPShellSet_t *parseECPFileForElement(const char *ecpFilePath, const char *element) 
{
    /* Open ECP file */
    FILE *fin = fopen(ecpFilePath, "r");
    if (!fin) {
        fprintf(stderr, "Unable to open ECP file\n");
        exit(EXIT_FAILURE);
    }

    /* How long is the atomic symbol? */
    size_t element_length = strlen(element);
    if (element_length != 1 && element_length != 2) {
        fprintf(stderr, "Incorrect element length: %zu\n", element_length);
        exit(EXIT_FAILURE);
    }

    /* Grab the atomic symbol in uppercase */
    char element_upper[7];
    strncpy(element_upper, element, element_length);
    element_upper[element_length + 0]='-'; 
    element_upper[element_length + 1]='E'; 
    element_upper[element_length + 2]='C'; 
    element_upper[element_length + 3]='P'; 
    element_upper[element_length + 4]='\0'; 
    ecp_to_upper(element_upper);

    /* First pass through the ECP file to count shells and primitives */

    /* Signal an sscanf failure */
    int fail_flag = 0;

    size_t n_shells = 0, n_prims_total = 0;

    /* The current line is in the target shell block */
    int in_target_block = 0;

    int max_L = 0;
    int nelec = 0;

    int nskip = 0;
    char line[512];

    while (fgets(line, sizeof(line), fin)) {
        /* Get the length of the line */
        size_t len = strlen(line);

        /* Skip comment lines (marked by '!') */
        char *p = line;
        while (*p && isspace((unsigned char) *p)) p++;
        if (*p == '!') continue;

        /* Remove trailing whitespace */
        while (len && isspace(line[len-1])) {
            len--;
            line[len] = '\0';
        }

        /* Skip blank lines */
        if (!len) {
            continue;
        }

        /* Skip contraction coefficients/exponents */
        if (nskip) {
            nskip--;
            continue;
        }

        /* Only parse lines with one, two, or three words */
        int nwords = ecp_count_words(line);
        if (nwords != 1 && nwords != 2 && nwords != 3) {
            continue; 
        }

        /* potential label */
        if (in_target_block && nwords == 2) {
            char shell_type[4], label[32];
            if (sscanf(line, "%3s %s", shell_type, label) != 2) {
                fail_flag = 1;
                break;
            }
            if (strstr(label, "0")) {
                /* Beginning of the next block - so we're done */
                break;
            } else {
                /* If we want to do something with shell_type, do it here */
                /* Format: s-f potential */
            }
            continue;
        }

        /* number of primitives */
        if (in_target_block && nwords == 1) {
            int nprim = 0;
            if (sscanf(line, "%d", &nprim) == 1) {
                /* Set nskip to skip parsing of the primitives */
                nskip = nprim;
                n_shells++; 
                n_prims_total += nprim; 
            } else {
                fail_flag = 1;
                break;
            }
            continue;
        }

        /* This is the first line of a new atom block */
        /* If we find the target element, set in_target_block, max_L, and nelec */
        if (!in_target_block && nwords == 3) {
            char block_atom[7];
            if (sscanf(line, "%7s %*s %*s", block_atom) == 1) {
                ecp_to_upper(block_atom);
                if (strcmp(block_atom, element_upper) == 0) {
                    in_target_block = 1;
                    if (sscanf(line, "%7s %i %i", block_atom, &max_L, &nelec) != 3) {
                        fail_flag = 1;
                        break;
                    }
                }
            } else {
                fail_flag = 1;
                break;
            }
        }
    }

    /* If this flag is not set, the desired element could not be found */
    if (!in_target_block) { 
        fclose(fin); 
        return NULL;
    }

    /* One of the sscanf calls failed */
    if (fail_flag) {
        fclose(fin); 
        fprintf(stderr, "ECP file parsing failed\n");
        exit(EXIT_FAILURE);
    }
 
    // Allocate output arrays
    uint64_t *shell_types  = (uint64_t*) malloc(n_shells * sizeof(uint64_t));
    uint64_t *num_prims    = (uint64_t*) malloc(n_shells * sizeof(uint64_t));
    uint64_t *prim_offsets = (uint64_t*) malloc(n_shells * sizeof(uint64_t));
    uint64_t *Ns           = (uint64_t*) malloc(n_prims_total * sizeof(uint64_t));
    double   *exponents    = (double*) malloc(n_prims_total * sizeof(double));
    double   *coefficients = (double*) malloc(n_prims_total * sizeof(double));

    if (!shell_types || !num_prims || !prim_offsets || !Ns || !exponents || !coefficients) {
        if (shell_types) free(shell_types); 
        if (num_prims) free(num_prims); 
        if (prim_offsets) free(prim_offsets);  
        if (Ns) free(Ns); 
        if (exponents) free(exponents); 
        if (coefficients) free(coefficients); 
        fclose(fin);
        exit(EXIT_FAILURE);
    }

    /* Second pass through the ECP file to populate shells and primitives */
    rewind(fin); 

    /* Indexing to populate these arrays */
    size_t nshell_index = 0;
    size_t nprimitive_index = 0;

    in_target_block = 0;
    int nprimitives_to_parse = 0;
    while (fgets(line, sizeof(line), fin)) {
        /* Get the length of the line */
        size_t len = strlen(line);

        /* Skip comment lines (marked by '!') */
        char *p = line;
        while (*p && isspace((unsigned char) *p)) p++;
        if (*p == '!') continue;

        /* Remove trailing whitespace */
        while (len && isspace(line[len-1])) {
            len--;
            line[len] = '\0';
        }

        /* Skip blank lines */
        if (!len) {
            continue;
        }

        /* Parse the contraction coefficients/exponents */
        if (nprimitives_to_parse && in_target_block) {
            double exp, coef;
            int N;
            char t1[64], t2[64], t3[64];
            if (sscanf(p, "%63s %63s %63s", t1, t2, t3) == 3) {
                N = strtol(t1, (char **) NULL, 10);
                exp = ecp_parse_float(t2);
                coef = ecp_parse_float(t3);

                Ns[nprimitive_index] = N;
                exponents[nprimitive_index] = exp;
                coefficients[nprimitive_index] = coef;
                nprimitive_index++;

                nprimitives_to_parse--;
                continue;
            } else {
                fail_flag = 1;
                break;
            }
        }

        /* Only parse lines with one, two, or three words */
        int nwords = ecp_count_words(line);
        if (nwords != 1 && nwords != 2 && nwords != 3) {
            continue; 
        }

        /* potential label */
        if (in_target_block && nwords == 2) {
            char shell_type[4], label[32];
            if (sscanf(line, "%3s %s", shell_type, label) != 2) {
                fail_flag = 1;
                break;
            }
            if (strstr(label, "0")) {
                /* Beginning of the next block - so we're done */
                break;
            } else {
                /* Format: s-f potential */
                uint64_t L = ecp_angular_momentum_type_to_L(shell_type[0]);
                shell_types[nshell_index] = L;
            }
            continue;
        }

        /* number of primitives */
        if (in_target_block && nwords == 1) {
            int nprim = 0;
            if (sscanf(line, "%d", &nprim) == 1) {
                num_prims[nshell_index] = nprim;
                if (nshell_index == 0) {
                    prim_offsets[nshell_index] = 0;
                } else {
                    prim_offsets[nshell_index] = prim_offsets[nshell_index - 1] + num_prims[nshell_index - 1];
                }
                nshell_index++;
                /* Set nprimitives_to_parse to the number of primitives */
                nprimitives_to_parse = nprim;
            } else {
                fail_flag = 1;
                break;
            }
            continue;
        }

        /* This is the first line of a new atom block */
        /* If we find the target element, set in_target_block */
        if (!in_target_block && nwords == 3) {
            char block_atom[7];
            if (sscanf(line, "%7s %*s %*s", block_atom) == 1) {
                ecp_to_upper(block_atom);
                if (strcmp(block_atom, element_upper) == 0) {
                    in_target_block = 1;
                }
            } else {
                fail_flag = 1;
                break;
            }
        }
    }

    fclose(fin);

    ECPShellSet_t* result = (ECPShellSet_t*) malloc(sizeof(ECPShellSet_t));

    if (!result || fail_flag) {
        if (result) free(result);
        free(shell_types); 
        free(num_prims); 
        free(prim_offsets); 
        free(Ns);
        free(exponents);
        free(coefficients);
        fprintf(stderr, "ECP file parsing failed\n");
        exit(EXIT_FAILURE);
    }

    result->n_shells = n_shells;
    result->n_elec = nelec;
    result->shell_types = shell_types;
    result->num_primitives = num_prims;
    result->primitive_offsets = prim_offsets;
    result->Ns = Ns;
    result->exponents = exponents;
    result->coefficients = coefficients;

    return result;
}

static void freeParsedECPFile(ECPShellSet_t *data) 
{
    if (data == NULL) {
        return;
    }
    if (data->shell_types) {
        free(data->shell_types);
    }
    if (data->num_primitives) {
        free(data->num_primitives);
    }
    if (data->primitive_offsets) {
        free(data->primitive_offsets);
    }
    if (data->Ns) {
        free(data->Ns);
    }
    if (data->exponents) {
        free(data->exponents);
    }
    if (data->coefficients) {
        free(data->coefficients);
    }
    free(data);
}

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_HELPER_ECP_PARSER */
