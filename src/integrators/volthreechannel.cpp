
// USE THIS 

// integrators/volthreechannel.cpp*
#include "integrators/volthreechannel.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"
#include "media/tc.h"


namespace pbrt {

STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

// VolPathIntegrator Method Definitions
void VolTCPathIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
    ReadLUT(color, sigmaaOutter, sigmaaInner);
}

Spectrum VolTCPathIntegrator::Li(const RayDifferential &r, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
	Spectrum L_3(0.0f);  //L_3 is the result of three spectrum

	for (int i = 0; i < 3 ; i++) { // compute per channel Li
        Spectrum L(0.f), beta(1.f);
        RayDifferential ray(r);
        bool specularBounce = false;
        int bounces = 0;
        Float etaScale = 1;
        for (bounces = 0;; ++bounces) {
            // Intersect _ray_ with scene and store intersection in _isect_
            SurfaceInteraction isect;
            bool foundIntersection = scene.Intersect(ray, &isect);  // compute the ray.tmax in this line, medium.sample use ray.tmax
            // Sample the participating medium, if present
            MediumInteraction mi;
            if (ray.medium && foundIntersection) {  // add foundintersection by WQ. But can not handle head in volume
                Spectrum omega;
				// Get albedo here, only tcmedium has albedo 
				if (const TCMedium * tcmedium = dynamic_cast<const TCMedium *>(ray.medium)) {
                    int flag = tcmedium->GetFlag();
                    omega = tcmedium->SampleKd(ray, sampler, arena, &mi, i, isect, scene, this->color, this->sigmaaOutter, this->sigmaaInner);
                } else {
                    omega = ray.medium->Sample(ray, sampler, arena, &mi, i);
                }
                beta *= omega;
            }
            if (beta.IsBlack()) break;

            // Handle an interaction with a medium or a surface
            if (mi.IsValid()) {
                // Terminate path if ray escaped or _maxDepth_ was reached
                if (bounces >= maxDepth) break;
                ++volumeInteractions;
                // Handle scattering at point in medium for volumetric path
                // tracer
                const Distribution1D *lightDistrib = lightDistribution->Lookup(mi.p);
                L += beta * UniformSampleOneLight(mi, scene, arena, sampler, this->color, this->sigmaaOutter, this->sigmaaInner, lightDistrib, i, true);
                Vector3f wo = -ray.d, wi;
                mi.phase->Sample_p(wo, &wi, sampler.Get2D());
                ray = mi.SpawnRay(wi);
                specularBounce = false;
            } else {
                ++surfaceInteractions;
                // Handle scattering at point on surface for volumetric path tracer
                // Possibly add emitted light at intersection
                if (bounces == 0 || specularBounce) {
                    // Add emitted light at path vertex or from the environment
                    if (foundIntersection)
                        L += beta * isect.Le(-ray.d);
                    else
                        for (const auto &light : scene.infiniteLights)
                            L += beta * light->Le(ray);
                }

                // Terminate path if ray escaped or _maxDepth_ was reached
                if (!foundIntersection || bounces >= maxDepth) break;

                // Compute scattering functions and skip over medium boundaries
                isect.ComputeScatteringFunctions(ray, arena, true);
                if (!isect.bsdf) {
                    ray = isect.SpawnRay(ray.d);
                    bounces--;
                    continue;
                }

                // Sample illumination from lights to find attenuated path
                // contribution
                const Distribution1D *lightDistrib =
                    lightDistribution->Lookup(isect.p);
                L += beta * UniformSampleOneLight(isect, scene, arena, sampler,
                                                  true, lightDistrib);

                // Sample BSDF to get new path direction
                Vector3f wo = -ray.d, wi;
                Float pdf;
                BxDFType flags;
                Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(),
                                                  &pdf, BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0.f) break;
                beta *= f * AbsDot(wi, isect.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);
                specularBounce = (flags & BSDF_SPECULAR) != 0;
                if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
                    Float eta = isect.bsdf->eta;
                    // Update the term that tracks radiance scaling for
                    // refraction depending on whether the ray is entering or
                    // leaving the medium.
                    etaScale *=
                        (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
                }
                ray = isect.SpawnRay(wi);
            }

            // Possibly terminate the path with Russian roulette
            // Factor out radiance scaling due to refraction in rrBeta.
            Spectrum rrBeta = beta * etaScale;
            if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
                Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
                if (sampler.Get1D() < q) break;
                beta /= 1 - q;
                DCHECK(std::isinf(beta.y()) == false);
            }
        }
        L_3[i] = L[i];
    }

    return L_3;
}

VolTCPathIntegrator *CreateVolTCPathIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    return new VolTCPathIntegrator(maxDepth, camera, sampler, pixelBounds,
                                 rrThreshold, lightStrategy);
}

}  // namespace pbrt
