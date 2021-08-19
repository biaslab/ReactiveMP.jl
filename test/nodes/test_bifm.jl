module BIFMNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "BIFMNode" begin

    @testset "Creation" begin
        node = make_node(BIFM, meta = BIFMMeta(ones(2,2), 2*ones(2,2), 3*ones(2,2)))

        @test functionalform(node)          === BIFM
        @test sdtype(node)                  === Deterministic()
        @test name.(interfaces(node))       === (:out, :in, :zprev, :znext)
        @test factorisation(node)           === ((1, 2, 3, 4), )
        @test metadata(node).A              ==  ones(2,2)
        @test metadata(node).B              ==  2*ones(2,2)
        @test metadata(node).C              ==  3*ones(2,2)

        node = make_node(BIFM, meta = 1)

        @test functionalform(node)          === BIFM
        @test sdtype(node)                  === Deterministic()
        @test name.(interfaces(node))       === (:out, :in, :zprev, :znext)
        @test factorisation(node)           === ((1, 2, 3, 4), )
        @test metadata(node)                === 1
    end
end
end