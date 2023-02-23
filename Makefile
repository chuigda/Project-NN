ifndef CC
	CC = gcc
endif
ifndef CFLAGS
	CFLAGS = -Wall -Wextra -Iinclude -g $(EXTRA_CFLAGS)
endif

define LOG
	@printf '\t%s\t%s\n' $1 $2
endef

define COMPILE
	$(call LOG,CC,$1)
	@$(CC) $(CFLAGS) $1 -fPIC -c -o $2
endef

HEADER_FILES = $(wildcard include/*.h)
IMPL_HEADER_FILES = $(wildcard include/impl/*.h)
SOURCE_FILES = $(wildcard src/*.c)
IMPL_SOURCE_FILES = $(wildcard src/impl/*.c)

OBJECT_FILES := $(patsubst src/%.c,%.o,$(SOURCE_FILES))
IMPL_OBJECT_FILES := $(patsubst src/impl/%.c,%.o,$(IMPL_SOURCE_FILES))

.PHONY: all libn2nn libn2nn-log
all: libn2nn

libn2nn: libn2nn-log libn2nn.so

libn2nn-log:
	@echo 'Building libn2nn.so'

libn2nn.so: $(IMPL_OBJECT_FILES) $(OBJECT_FILES)
	$(call LOG,LD,libn2nn.so)
	@$(CC) \
		$(CFLAGS) $(IMPL_OBJECT_FILES) $(OBJECT_FILES) \
		-fPIC -shared -o libn2nn.so

%.o: src/impl/%.c $(IMPL_HEADER_FILES)
	$(call COMPILE,$<,$@)

%.o: src/%.c $(HEADER_FILES) $(IMPL_HEADER_FILES)
	$(call COMPILE,$<,$@)

define BUILD_TEST_ITEM
	$(call LOG,BUILD,$1)
	@$(CC) $(CFLAGS) $2 -L. -ln2nn -lm -o $1
endef

define RUN_TEST_ITEM
	@printf '\tTEST\t%s\t\t\tCASE %s/%s\t' $1 $2 $3
	@LD_LIBRARY_PATH=. ./$1.bin $2 > /dev/null
	@printf 'PASS\n'
endef

.PHONY: test test-log
test: libn2nn \
	test-log \
	test_arena \
	test_deriv \
	test_neuron \
	test_fnn

test-log:
	@echo 'Running tests'

.PHONY: test_arena
test_arena: test_arena.bin
	$(call RUN_TEST_ITEM,test_arena,1,1)

.PHONY: test_deriv
test_deriv: test_deriv.bin
	$(call RUN_TEST_ITEM,test_deriv,1,1)

.PHONY: test_neuron
test_neuron: test_neuron.bin
	$(call RUN_TEST_ITEM,test_neuron,1,2)
	$(call RUN_TEST_ITEM,test_neuron,2,2)

.PHONY: test_fnn
test_fnn: test_fnn.bin
	$(call RUN_TEST_ITEM,test_fnn,1,1)

%.bin: test/%.c libn2nn.so
	$(call BUILD_TEST_ITEM,$@,$<)

.PHONY: clean
clean:
	rm -f *.o
	rm -f *.a
	rm -f *.so
	rm -f test_*.bin*
	rm -rf html
	rm -rf latex
