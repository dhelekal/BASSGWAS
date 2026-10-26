# Copyright (C) 2026 David Helekal
# SPDX-License-Identifier: GPL-3.0-only
using Test
using BASSGWAS
using LinearAlgebra

function record_block!(data, lo, hi, worker)
    # Reuse worker-owned scratch across blocks; its contents identify the owner.
    buffer = data.buffers[worker]
    fill!(buffer,worker)
    data.worker_calls[worker] += 1
    @inbounds for i in lo:hi
        data.visits[i] += 1
        data.owners[i] = buffer[1]
    end
    data.calls[lo] = hi
    return data.failures[lo]
end

function block_test_data(n, workers)
    return (;visits=zeros(Int,n),owners=zeros(Int,n),calls=zeros(Int,n),
        failures=zeros(Int,n),worker_calls=zeros(Int,workers),
        buffers=[zeros(Int,1) for _ in 1:workers])
end

@testset "Contiguous block coverage and logical worker indices" begin
    for n in (0,1,255,256,257,1027), requested in (1,3,Threads.nthreads()+2)
        data = block_test_data(n,requested)
        result = BASSGWAS.foreach_block!(record_block!,data,n;
            blocksize=256,workers=requested)
        nw = min(requested,Threads.nthreads(),cld(n,256))
        @test result == 0
        @test all(==(1),data.visits)
        @test issorted(data.owners)
        @test unique(data.owners) == collect(1:nw)
        @test data.worker_calls == [count(==(worker),data.owners[1:256:n]) for worker in 1:requested]
        @test findall(!iszero,data.calls) == collect(1:256:n)
        @test data.calls[1:256:n] == min.(collect(256:256:(cld(n,256)*256)),n)

        serial = block_test_data(n,1)
        @test BASSGWAS.foreach_block!(record_block!,serial,n;
            blocksize=256,workers=1) == 0
        @test serial.visits == data.visits && serial.calls == data.calls
        @test serial.owners == ones(Int,n)
    end
end

@testset "Block failure selection, stopping and recovery" begin
    # Three blocks per logical worker. Distinct codes deliberately run opposite
    # to worker order: the result must not be a minimum-code reduction.
    for requested in (1,3)
        nw = min(requested,Threads.nthreads())
        n = 12*nw
        data = block_test_data(n,nw)
        # Fail worker 1 in its second block; fail later workers in their first.
        data.failures[5] = 90
        for worker in 2:nw
            data.failures[12*(worker-1)+1] = 90-worker
        end
        @test BASSGWAS.foreach_block!(record_block!,data,n;
            blocksize=4,workers=requested) == 90
        @test data.visits[1:8] == ones(Int,8)
        @test data.visits[9:12] == zeros(Int,4)
        for worker in 2:nw
            lo = 12*(worker-1)+1
            @test data.visits[lo:lo+3] == ones(Int,4)
            @test data.visits[lo+4:lo+11] == zeros(Int,8)
        end
        @test data.worker_calls == [2;ones(Int,nw-1)]

        fill!(data.failures,0)
        fill!(data.visits,0)
        @test BASSGWAS.foreach_block!(record_block!,data,n;
            blocksize=4,workers=requested) == 0
        @test all(==(1),data.visits)

        # A successful earlier worker must not hide a later worker's failure.
        data.failures[n-3] = -7
        @test BASSGWAS.foreach_block!(record_block!,data,n;
            blocksize=4,workers=requested) == -7
    end
end

function typed_dot_block!(data, lo, hi, worker)
    # Return a failure instead of throwing inside a Polyester callback.
    typeof(data.A) === data.matrix_type || return 1
    typeof(data.out) === data.output_type || return 2
    mul!(view(data.out,lo:hi),transpose(view(data.A,:,lo:hi)),data.x)
    return 0
end

@testset "Original array types and BLAS dispatch" begin
    original_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        storage = reshape(sin.(collect(1:8*1027)),8,1027)
        x = collect(1.0:8.0)
        for strided in (false,true)
            A = strided ? view(storage,1:2:8,:) : storage
            output_storage = fill(NaN,3,1027)
            out = strided ? view(output_storage,2,:) : zeros(1027)
            data = (;A,out,x=x[1:size(A,1)],matrix_type=typeof(A),output_type=typeof(out))
            @test BASSGWAS.foreach_block!(typed_dot_block!,data,1027;blocksize=256) == 0
            @test out ≈ transpose(A)*data.x
            @test all(isnan,view(output_storage,1,:)) && all(isnan,view(output_storage,3,:))
        end
    finally
        BLAS.set_num_threads(original_threads)
    end
end

function independent_block_run(code)
    data = block_test_data(1027,Threads.nthreads())
    data.failures[end-2] = code # First index of the partial final block.
    status = BASSGWAS.foreach_block!(record_block!,data,1027;blocksize=256)
    return status,data.visits,data.owners
end

@testset "Independent concurrent block runners" begin
    expected = [independent_block_run(code) for code in (0,17)]
    tasks = [Threads.@spawn independent_block_run(code) for code in (0,17)]
    @test fetch.(tasks) == expected
end

function no_op_block!(data, lo, hi, worker)
    return 0
end

function block_allocations(n, workers)
    BASSGWAS.foreach_block!(no_op_block!,nothing,n;blocksize=16,workers)
    return @allocated BASSGWAS.foreach_block!(no_op_block!,nothing,n;blocksize=16,workers)
end

@testset "Block arguments and bounded status allocations" begin
    for (n,blocksize,workers) in ((-1,1,1),(0,0,1),(1,-1,1),(0,1,0),(1,1,-1))
        @test_throws ArgumentError BASSGWAS.foreach_block!(no_op_block!,nothing,n;blocksize,workers)
    end
    # The block callback must not run for empty input.
    @test BASSGWAS.foreach_block!(nothing,nothing,0;blocksize=1) == 0
    @test block_allocations(16384,1) == 0
    for workers in (1,3)
        small = block_allocations(4096,workers)
        large = block_allocations(16384,workers)
        @test small == large
        @test large <= 512+64*min(workers,Threads.nthreads())
    end
end
