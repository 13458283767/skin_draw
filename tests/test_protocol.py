import pytest

from skin_draw.protocol import BlockParser


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 17, 256])
@pytest.mark.parametrize("ending", [b"", b"END\r\n"])
def test_incremental_rows_optional_end_and_multiple_blocks(chunk_size, ending):
    data = b"BLOCK_1,2,3:\r\n0,1024,4095\r\n1,2,3\r\n" + ending + b"BLOCK_14,1,1:\r\n2048\r\n" + ending
    parser = BlockParser()
    blocks = []
    for offset in range(0, len(data), chunk_size):
        blocks.extend(parser.feed(data[offset:offset + chunk_size]))
    assert [b.block_id for b in blocks] == [1, 14]
    assert blocks[0].values.tolist() == [[0, 1024, 4095], [1, 2, 3]]
    assert parser.errors == 0


@pytest.mark.parametrize("bad", [b"1,,3\r\n", b"1,2\r\n", b"1,-2,3\r\n", b"1,4096,3\r\n",
                               b"a,2,3\r\n", b"END\r\n", b"x" * 600 + b"\r\n"])
def test_bad_block_does_not_shift_rows_and_next_header_resynchronizes(bad):
    parser = BlockParser()
    blocks = parser.feed(b"BLOCK_1,2,3:\r\n" + bad + b"4,5,6\r\nBLOCK_2,1,1:\r\n1234\r\nEND\r\n")
    assert [block.block_id for block in blocks] == [2]
    assert parser.errors >= 1


@pytest.mark.parametrize("header", [b"BLOCK_0,1,1:", b"BLOCK_17,1,1:", b"BLOCK_1,99,1:", b"BLOCK_1,0,1:", b"BLOCK_1,1,1"])
def test_bad_headers_are_rejected(header):
    parser = BlockParser()
    assert parser.feed(header + b"\r\n1\r\nEND\r\n") == []
    assert parser.errors > 0

