import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration.StandardSequences.Services_cff')
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('RecoTracker.Configuration.RecoTracker_cff')
#process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.GlobalTag.globaltag =  cms.string("106X_mc2017_realistic_v10")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL17MiniAODv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/00000/00623789-C012-474B-8860-3C18397A464F.root")
)

process.demo = cms.EDAnalyzer('DemoAnalyzer',
packed = cms.InputTag("packedGenParticles"),
pruned = cms.InputTag("prunedGenParticles"),
tracks = cms.untracked.InputTag('packedPFCandidates'),
jets = cms.untracked.InputTag('slimmedJets'),
primaryVertices = cms.untracked.InputTag('offlineSlimmedPrimaryVertices'),
losttracks = cms.untracked.InputTag('lostTracks', '', "PAT"),
TrackPtCut = cms.untracked.double(1.0)
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string("test_output.root"),
)

process.p = cms.Path(process.demo)
