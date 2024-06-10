using Test: @inferred
using Test: @test
using Test: @test_broken
using Test: @testset
using ZebrafishHMM2023: chunks
using ZebrafishHMM2023: equal_partition
using ZebrafishHMM2023: find_repeats
using ZebrafishHMM2023: makechunks
using ZebrafishHMM2023: split_into_repeated_subsequences

@testset "chunks" begin
    @test chunks(1:54, 5) == [1:11, 12:22, 23:33, 34:44, 45:54]
    @test chunks(2:55, 5) == [2:12, 13:23, 24:34, 35:45, 46:55]
    @test_broken length(chunks(1:32714, 1635)) == 1635
end

# @testset "makechunks" begin
#     @test makechunks(1:54, 5) == [1:11, 12:22, 23:33, 34:44, 45:54]
#     @test makechunks(2:55, 5) == [2:12, 13:23, 24:34, 35:45, 46:55]
#     @test length(makechunks(1:32714, 1635)) == 1635
# end

@testset "equal_partition" begin
    @test equal_partition(1:54, 5) == [1:11, 12:22, 23:32, 33:43, 44:54]
    @test equal_partition(2:55, 5) == [2:12, 13:23, 24:33, 34:44, 45:55]
    @test length(equal_partition(1:32714, 1635)) == 1635
    @test reduce(vcat, equal_partition(1:32714, 1635)) == 1:32714
end

@testset "split_into_repeated_subsequences" begin
    @test split_into_repeated_subsequences([1,1,2,2]) == [[1,1],[2,2]]
    seq = reduce(vcat, [fill(rand(1:3), rand(1:20)) for _ = 1:20])
    subseqs = split_into_repeated_subsequences(seq)
    @test reduce(vcat, subseqs) == seq
    @test all(map(length ∘ unique, subseqs) .== 1)
    for (s1, s2) = zip(subseqs[1:end - 1], subseqs[2:end])
        @test first(s1) ≠ first(s2)
    end
end

@testset "find_repeats" begin
    @test find_repeats([1,1,2,2]) == [1:2, 3:4]
    @test find_repeats(Int[]) == [1:0]
    @test find_repeats([1]) == [1:1]
    seq = reduce(vcat, [fill(rand(1:3), rand(1:20)) for _ = 1:20])
    reps = find_repeats(seq)
    @test reduce(vcat, reps) == 1:length(seq)
    for r = reps
        @test allequal(seq[r])
        @test all(==(seq[first(r)]), seq[r])
        if first(r) > 1
            @test seq[first(r) - 1] ≠ seq[first(r)]
        end
        if last(r) < length(seq)
            @test seq[last(r) + 1] ≠ seq[last(r)]
        end
    end
end
