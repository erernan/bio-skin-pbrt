// ABORT!!!!
// integrators/volpath.cpp*
#include "integrators/skinpath.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"
#include "media/homogeneous.h"
#include "materials/albedo.h"

namespace pbrt {

STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

// VolPathIntegrator Method Definitions
void SkinPathIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

void SkinPathIntegrator::FindCloest(Spectrum &color, Spectrum &sigmaa,
                                    int32_t flag) const {
    Vector3f alb = {color[0], color[1], color[2]};
    int maxid = 0;
    Float minDist = MaxFloat;
    for (int i = 0; i < this->color.size(); i++) {
        Float currentDist = abs(alb[0] - this->color[i][0]) +
                            abs(alb[1] - this->color[i][1]) +
                            abs(alb[2] - this->color[i][2]);
        if (currentDist < minDist) {
            minDist = currentDist;  
            maxid = i;
        }
	}
    Vector3f sigmaVector = (flag == 1 ? this->sigmaaOutter[maxid] : this->sigmaaInner[maxid]);
    Float sigmaArray[3] = {sigmaVector[0], sigmaVector[1], sigmaVector[2]};
    sigmaa = Spectrum::FromRGB(sigmaArray);
}

Spectrum SkinPathIntegrator::Li(const RayDifferential &r, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;
    Spectrum Kd {0.0};
	// First I need the material
	SurfaceInteraction isectAlbedo;
    Ray rayAlbedo = ray;
    const AlbedoMaterial *albedo = nullptr;
    bool albedoIntersection = scene.Intersect(rayAlbedo, &isectAlbedo);
    if (albedoIntersection) {
        if (albedo = dynamic_cast<const AlbedoMaterial *>(
                isectAlbedo.primitive->GetMaterial())) {
        }
    }
    for (bounces = 0;; ++bounces) {
        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect); // compute the ray.tmax in this line, medium.sample use ray.tmax
        // Sample the participating medium, if present
        MediumInteraction mi;
		//now we have the LUT in memroy, then we need to compute the corresponding sigma according to albedo Kd
		// first get the flag by ray.meidum
        Spectrum sigmas;
        int32_t flag;
		if (ray.medium) {
			if (const HomogeneousMedium *medium =
				dynamic_cast<const HomogeneousMedium *>(ray.medium))
			{
                flag = medium->GetFlag();
                            //std::cout << flag << std::endl;
			}
			// now we have flag, then we need to find the cloest LUT value by Kd
            FindCloest(Kd, sigmas, flag);
            // beta *=dynamic_cast<const HomogeneousMedium *>(ray.medium)->SampleKd(ray, sampler, arena, &mi, sigmas);
            beta *=dynamic_cast<const HomogeneousMedium *>(ray.medium)->SampleKd(ray, sampler, arena, &mi, sigmas);
		}
        if (beta.IsBlack()) break;

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;

            ++volumeInteractions;
			// compute mi uv according to flag
			// if flag is 1 or 2, compute uv
            Point2f uv {0.0,0.0};
            SurfaceInteraction isect_reverse;
            if (flag != 0) {
				// we might need 2 other intersection to compute uv. However, we have the second intersection at line 80   -------/----   <----- 2 interscetions
				// So we just spwan 1 ray and compute the intersection.                                                    ------/-----    
				Ray ray_reverse = Ray(ray.o, -ray.d);
                bool foundReverse = scene.Intersect(ray_reverse, &isect_reverse);
                Point2f uv_front = isect.uv; 
				Point2f uv_reverse = isect_reverse.uv;
				// interpolate to get uv at mi
                Float bilinear = ray.tMax / (ray_reverse.tMax + ray.tMax);
				if (bilinear > 0.9999){
                    uv = uv_reverse;
                } else if (bilinear < 0.0001) {
					uv = uv_front;
                } else {
                    uv = bilinear * uv_reverse + (1.0 - bilinear) * uv_front;
				}
			}
			
			// After get uv, it's needed to compute albedo according to uv;
            isect.uv = uv;
            Kd = albedo->GetAlbedo(&isect);
			
            // Handle scattering at point in medium for volumetric path tracer
            const Distribution1D *lightDistrib =
                lightDistribution->Lookup(mi.p);
            L += beta * UniformSampleOneLight(mi, scene, arena, sampler, true,
                                              lightDistrib);

            Vector3f wo = -ray.d, wi;
            mi.phase->Sample_p(wo, &wi, sampler.Get2D());
            ray = mi.SpawnRay(wi);
            
            specularBounce = false;
        } else {
            ++surfaceInteractions;
            // Handle scattering at point on surface for volumetric path tracer

			// Handle albedo compute at surface interaction
            if (albedo != nullptr) {
                Kd = albedo->GetAlbedo(&isect);
            }


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
            Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                              BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0.f) break;
            beta *= f * AbsDot(wi, isect.shading.n) / pdf;
            DCHECK(std::isinf(beta.y()) == false);
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
                Float eta = isect.bsdf->eta;
                // Update the term that tracks radiance scaling for refraction
                // depending on whether the ray is entering or leaving the
                // medium.
                etaScale *=
                    (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
            }
            ray = isect.SpawnRay(wi);

            // Account for attenuated subsurface scattering, if applicable
            if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
                // Importance sample the BSSRDF
                SurfaceInteraction pi;
                Spectrum S = isect.bssrdf->Sample_S(
                    scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
                DCHECK(std::isinf(beta.y()) == false);
                if (S.IsBlack() || pdf == 0) break;
                beta *= S / pdf;

                // Account for the attenuated direct subsurface scattering
                // component
                L += beta *
                     UniformSampleOneLight(pi, scene, arena, sampler, true,
                                           lightDistribution->Lookup(pi.p));

                // Account for the indirect subsurface scattering component
                Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(),
                                               &pdf, BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0) break;
                beta *= f * AbsDot(wi, pi.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);
                specularBounce = (flags & BSDF_SPECULAR) != 0;
                ray = pi.SpawnRay(wi);
            }
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
    ReportValue(pathLength, bounces);
    //std::cout << "L:" << L[0] << " " << L[1] << " " << L[2] << std::endl;
	return L;
}

SkinPathIntegrator *CreateSkinPathIntegrator(
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
    return new SkinPathIntegrator(maxDepth, camera, sampler, pixelBounds,
                                 rrThreshold, lightStrategy);
}

}  // namespace pbrt
