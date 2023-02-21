ifndef CC
	CC = gcc
endif
ifndef CFLAGS
	CFLAGS = -Wall -Wextra -Iinclude -O2 -g
endif

define LOG
	@printf '\t%s\t%s\n' $1 $2
endef

define COMPILE
	$(call LOG,CC,$1)
	@$(CC) $(CFLAGS) $1 -fPIC -c -o $2
endef

HEADER_FILES = $(wildcard include/*.h)
SOURCE_FILES = $(wildcard src/*.c)

OBJECT_FILES := $(patsubst src/%.c,%.o,$(SOURCE_FILES))

libn2nn.so: $(OBJECT_FILES)
	$(call LOG,LINK,libn2nn.so)
	@$(CC) $(CFLAGS) -fPIC -shared $(SOURCE_FILES) -o libn2nn.so

%.o: src/%.c $(SOURCE_FILES)
	$(call COMPILE,$<,$@)

define BUILD_TEST_ITEM
	$(call LOG,BUILD,$1)
	@$(CC) $(CFLAGS) $2 -L. -ln2nn -lm -o $1
endef

define RUN_TEST_ITEM
	@printf '\tTEST\t%s\t\t\t' $1
	@LD_LIBRARY_PATH=. ./$1 > /dev/null
	@if [ $$? -eq 0 ]; then printf 'PASS\n'; else printf 'FAIL\n'; fi
endef

.PHONY: test
test: test_neuron.bin
	$(call RUN_TEST_ITEM,test_neuron.bin)

%.bin: test/%.c libn2nn.so
	$(call BUILD_TEST_ITEM,test_neuron.bin,test/test_neuron.c)

.PHONY: clean
clean:
	rm -f *.o
	rm -f *.a
	rm -f *.so
	rm -f test_*.bin*
