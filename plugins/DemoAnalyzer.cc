// -*- C++ -*-
//
// Package:    Demo/DemoAnalyzer
// Class:      DemoAnalyzer
//
/**\class DemoAnalyzer DemoAnalyzer.cc Demo/DemoAnalyzer/plugins/DemoAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Nikhilesh Venkatasubramanian
//         Created:  Tue, 16 Apr 2024 18:10:25 GMT
//
//

//
// constructors and destructor

#include "dispV/dispVAnalyzer/interface/DemoAnalyzer.h"
#include <iostream>

DemoAnalyzer::DemoAnalyzer(const edm::ParameterSet& iConfig):

	theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
	TrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
	PVCollT_ (consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("primaryVertices"))),
  	LostTrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("losttracks"))),
	jet_collT_ (consumes<edm::View<reco::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
	prunedGenToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("pruned"))),
  	packedGenToken_(consumes<edm::View<pat::PackedGenParticle> >(iConfig.getParameter<edm::InputTag>("packed"))),
	TrackPtCut_(iConfig.getUntrackedParameter<double>("TrackPtCut"))
{
	edm::Service<TFileService> fs;	
	//usesResource("TFileService");
   	tree = fs->make<TTree>("tree", "tree");
}

DemoAnalyzer::~DemoAnalyzer() {
}

//bool DemoAnalyzer::isAncestor(const reco::Candidate* ancestor, const reco::Candidate * particle)
//{
////particle is already the ancestor
//        if(ancestor == particle ) return true;
//
////otherwise loop on mothers, if any and return true if the ancestor is found
//        for(size_t i=0;i< particle->numberOfMothers();i++)
//        {
//                if(isAncestor(ancestor,particle->mother(i))) return true;
//        }
////if we did not return yet, then particle and ancestor are not relatives
//        return false;
//}

std::optional<std::tuple<float, float, float>> DemoAnalyzer::isAncestor(const reco::Candidate* ancestor, const reco::Candidate* particle)
{
    // Particle is already the ancestor
    if (ancestor == particle) {
        // Use NaN values to indicate that this is the ancestor but we are not returning its vertex
        return std::make_optional(std::make_tuple(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
    }

    // Otherwise, loop on mothers, if any, and check for the ancestor in the next level up
    for (size_t i = 0; i < particle->numberOfMothers(); i++) {
        auto result = isAncestor(ancestor, particle->mother(i));
        if (result) {
            // If we found a NaN tuple, it means this particle is the child of the ancestor
            if (std::isnan(std::get<0>(*result))) {
                // So, return this particle's vertex since it's the direct descendant
                return std::make_optional(std::make_tuple(particle->vx(), particle->vy(), particle->vz()));
            } else {
                // Otherwise, keep passing up the found vertex coordinates
                return result;
            }
        }
    }

    // If we did not return yet, then particle and ancestor are not relatives
    return std::nullopt;  // Return an empty optional if no ancestor found
}


//
// member functions
//

// ------------ method called for each event  ------------
void DemoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace pat;

  BHadron_pt.clear();
  BHadron_eta.clear();
  BHadron_phi.clear();
  BHadron_SVx.clear();
  BHadron_SVy.clear();
  BHadron_SVz.clear(); 
  nBHadrons.clear();
  nBDaughters.clear();
  BDaughters_pt.clear();
  BDaughters_eta.clear();
  BDaughters_phi.clear();

  ntrks.clear();
  trk_ip2d.clear();
  trk_ip3d.clear();
  trk_ip2dsig.clear();
  trk_ip3dsig.clear();
  trk_pt.clear();
  trk_eta.clear();
  trk_phi.clear();

  njets.clear();
  jet_pt.clear();
  jet_eta.clear();
  jet_phi.clear();


  Handle<PackedCandidateCollection> patcan;
  Handle<PackedCandidateCollection> losttracks;
  Handle<edm::View<reco::Jet> > jet_coll;
  Handle<edm::View<reco::GenParticle> > pruned;
  Handle<edm::View<pat::PackedGenParticle> > packed;
  Handle<reco::VertexCollection> pvHandle;

  std::vector<reco::Track> alltracks;

  iEvent.getByToken(TrackCollT_, patcan);
  iEvent.getByToken(LostTrackCollT_, losttracks);
  iEvent.getByToken(jet_collT_, jet_coll);
  iEvent.getByToken(prunedGenToken_,pruned);
  iEvent.getByToken(packedGenToken_,packed);
  iEvent.getByToken(PVCollT_, pvHandle);

  const auto& theB = &iSetup.getData(theTTBToken);
  reco::Vertex pv = (*pvHandle)[0];

  GlobalVector direction(1,0,0);
  direction = direction.unit();


  for (auto const& itrack : *patcan){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
               alltracks.push_back(tmptrk);
           }
       }
   }

   for (auto const& itrack : *losttracks){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
               alltracks.push_back(itrack.pseudoTrack());
           }
       }
   }

   int njet = 0;
   for (auto const& ijet: *jet_coll){
   	jet_pt.push_back(ijet.pt());
	jet_eta.push_back(ijet.eta());
	jet_phi.push_back(ijet.phi());
	njet++;
   }
   njets.push_back(njet);
   
   int ntrk = 0;
   for (const auto& track : alltracks) {
	reco::TransientTrack t_trk = (*theB).build(track);
	if (!(t_trk.isValid())) continue;
	Measurement1D ip2d = IPTools::signedTransverseImpactParameter(t_trk, direction, pv).second;
	Measurement1D ip3d = IPTools::signedImpactParameter3D(t_trk, direction, pv).second;	

   	trk_ip2d.push_back(ip2d.value());
	trk_ip3d.push_back(ip3d.value());
	trk_ip2dsig.push_back(ip2d.significance());
	trk_ip3dsig.push_back(ip3d.significance());
	trk_pt.push_back(track.pt());
	trk_eta.push_back(track.eta());
	trk_phi.push_back(track.phi());
	ntrk++;
   }
   ntrks.push_back(ntrk);

   int nbhads = 0;
   for(size_t i=0; i< pruned->size();i++){
   	if((std::abs((*pruned)[i].pdgId())/100)%10 == 5 || (std::abs((*pruned)[i].pdgId())/1000)%10 == 5){
		nbhads++;
        	const Candidate * bHadron = &(*pruned)[i];
		BHadron_pt.push_back(bHadron->pt());
		BHadron_eta.push_back(bHadron->eta());
		BHadron_phi.push_back(bHadron->phi());
		int nbdaughters = 0;
                for(size_t j=0; j< packed->size();j++){
//get the pointer to the first survied ancestor of a given packed GenParticle in the prunedCollection
                                const Candidate * motherInPrunedCollection = (*packed)[j].mother(0) ;
                                if(motherInPrunedCollection != nullptr){
					auto SV = isAncestor(bHadron, motherInPrunedCollection);
					if(SV.has_value()){
						auto [vx, vy, vz] = *SV;
						if (!std::isnan(vx) && !std::isnan(vy) && !std::isnan(vz)) {
							nbdaughters++;
							BDaughters_pt.push_back((*packed)[j].pt());
							BDaughters_eta.push_back((*packed)[j].eta());
							BDaughters_phi.push_back((*packed)[j].phi());
							BHadron_SVx.push_back(vx);
							BHadron_SVy.push_back(vy); //Here each vertex coordinate will correspond to the vertex of the bhad
							BHadron_SVz.push_back(vz); //desc it thinks is coming from this corresponding daugther in the loop
						}
					}
                                }
                        }
		nBDaughters.push_back(nbdaughters);
                }

        }
   nBHadrons.push_back(nbhads);

   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void DemoAnalyzer::beginJob() {
	tree->Branch("nBHadrons", &nBHadrons);
	tree->Branch("BHadron_pt", &BHadron_pt);
	tree->Branch("BHadron_eta", &BHadron_eta);
	tree->Branch("BHadron_phi", &BHadron_phi);
	tree->Branch("BHadron_SVx", &BHadron_SVx);
	tree->Branch("BHadron_SVy", &BHadron_SVy);
	tree->Branch("BHadron_SVz", &BHadron_SVz);
	tree->Branch("nBDaughters", &nBDaughters);
	tree->Branch("BDaughters_pt", &BDaughters_pt);
	tree->Branch("BDaughters_eta", &BDaughters_eta);
	tree->Branch("BDaughters_phi", &BDaughters_phi);

	tree->Branch("nTrks", &ntrks);
	tree->Branch("trk_ip2d", &trk_ip2d);
	tree->Branch("trk_ip3d", &trk_ip3d);
	tree->Branch("trk_ip2dsig", &trk_ip2dsig);
        tree->Branch("trk_ip3dsig", &trk_ip3dsig);
	tree->Branch("trk_pt", &trk_pt);
	tree->Branch("trk_eta", &trk_eta);
	tree->Branch("trk_phi", &trk_phi);

	tree->Branch("nJets", &njets);
	tree->Branch("jet_pt", &jet_pt);
        tree->Branch("jet_eta", &jet_eta);
        tree->Branch("jet_phi", &jet_phi);

}


// ------------ method called once each job just after ending the event loop  ------------
void DemoAnalyzer::endJob() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DemoAnalyzer);
