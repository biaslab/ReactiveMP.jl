export rule

@rule BIFMHelper(:out, Marginalisation) (q_in::Any, ) = MarginalDistribution(q_in)