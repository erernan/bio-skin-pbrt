// ABORT

// media/heterogeneous.cpp*
#include "media/heterogeneous.h"
#include "sampler.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"
#include "scene.h"
#include "memory.h"

namespace pbrt {
STAT_RATIO("Media/Grid steps per Tr() call", nTrSteps, nTrCalls);


// compute ��s at a point
Spectrum HeterogeneousMedium::MiuS(const Point3f &p) const { 
    return this->sigma_s;
}

void HeterogeneousMedium::FindCloest(Spectrum &color, Spectrum &sigmaa,
                                     int32_t flag,
                                     const std::vector<Vector3f> &colorvec,
                                     const std::vector<Vector3f> &sig_aouter,
                                     const std::vector<Vector3f> &sig_ainner) const{
    Vector3f alb = {color[0], color[1], color[2]};
    int maxid = 0;
    Float minDist = MaxFloat;
    for (int i = 0; i < colorvec.size(); i++) {
        Float currentDist = abs(alb[0] - colorvec[i][0]) +
                            abs(alb[1] - colorvec[i][1]) +
                            abs(alb[2] - colorvec[i][2]);
        if (currentDist < minDist) {
            minDist = currentDist;
            maxid = i;
        }
    }
    Vector3f sigmaVector = (flag == 1 ? sig_aouter[maxid] : sig_ainner[maxid]);
    Float sigmaArray[3] = {sigmaVector[0], sigmaVector[1], sigmaVector[2]};
    sigmaa = Spectrum::FromRGB(sigmaArray);
}

// compute Tr by ratio tracking. If not using Russian roulette, there will be too mush sample, and pbrt can not generate too much sample(>1000), so Tr
// estimate is underestimate. Thus, maybe the three channel integrator is more reasonalbe
Spectrum HeterogeneousMedium::Tr(const Ray &ray, Sampler &sampler, int cha) const {
    ProfilePhase _(Prof::MediumTr);
	// init tmin and tmax
    Float tmin = 0.0;
    Float tmax = INFINITY;
	// normalize ray
    Ray rayNormalize = Ray(ray.o, Normalize(ray.d), ray.tMax * ray.d.Length());
    
	// find intersection? Robustness? Need not to intersect, the tmax was compute before
    tmax = rayNormalize.tMax;
    Spectrum Tr = Spectrum(1.0);
    Float t = tmin;
    while (true) {
        ++nTrSteps;
        t -= std::log(1 - sampler.Get1D()) * (1 / maxT);
        if (t >= tmax) break;
        Spectrum miuT = sigma_t;
        //Spectrum miuT = MiuT(ray(t));
        Tr *= Spectrum(1.0) - miuT * (1 / maxT);
        // when transmittance gets low, start applying Russian roulette to terminate sampling.
        const Float rrThreshold = 0.0001;
        if (Tr[0] < rrThreshold) {
            Float q = std::max((Float).05, 1 - Tr[0]);
            if (sampler.Get1D() < q) return 0;
            Tr /= 1 - q;
        }
    }
    return Spectrum(Tr);

	//ProfilePhase _(Prof::MediumTr);
 //   CoefficientSpectrum<3> tr =
	//	Exp(-sigma_t * std::min(ray.tMax * ray.d.Length(), MaxFloat));
 //   return tr;
}

Spectrum HeterogeneousMedium::Tr(
    const Ray &ray, Sampler &sampler, const Scene &scene,
    SurfaceInteraction &isect, const std::vector<Vector3f> &colorvec,
    const std::vector<Vector3f> &sig_aouter,
    const std::vector<Vector3f> &sig_ainner) const {

    ProfilePhase _(Prof::MediumTr);
    Spectrum Tr = Spectrum(1.0);
    Float t = 0.0;
    // Optimize. Only need one intersection per Tr and sample. Because ray.d doesn't change during compute
    Ray ray_reverse = Ray(ray(ray.tMax * 0.5), -ray.d);  // Ray origin is the middle of ray
    SurfaceInteraction isect_reverse;
    bool foundReverse = scene.Intersect(ray_reverse, &isect_reverse);
    // Get uv onece
    Point2f uv_reverse, uv_front;
    uv_front = isect.uv;
    uv_reverse = isect_reverse.uv;

    while (true) {
        Float dist = -std::log(1 - sampler.Get1D()) * (1.0 / maxT);
        t += std::min(dist / ray.d.Length(), ray.tMax);
        if (t >= ray.tMax) break;
        // interpolate to get uv at ray(t)
        Point2f uv{0.0, 0.0};
        Float bilinear = (ray.tMax - t) / (ray.tMax * 0.5 + ray_reverse.tMax);
        // Float bilinear = 1.0;
        if (bilinear > 0.99) {
            uv = uv_reverse;
        } else if (bilinear <= 0.01) {
            uv = uv_front;
        } else {
            uv = bilinear * uv_reverse + (1.0 - bilinear) * uv_front;
        }
        // Get albedo value
        SurfaceInteraction isect_albedo;
        isect_albedo.uv = uv;
        Spectrum albedo = Kd->Evaluate(isect_albedo);
        // Albedo2sigma
        Spectrum sigma_a_albedo;
        FindCloest(albedo, sigma_a_albedo, this->GetFlag(), colorvec,
                   sig_aouter, sig_ainner);

        Spectrum sigma_t = (sigma_a_albedo * scale + sigma_s);

        Tr *= Spectrum(1.0) - sigma_t * (1 / maxT);

        // Added after book publication: when transmittance gets low,
        // start applying Russian roulette to terminate sampling.
        const Float rrThreshold = .0001;
        if (Tr[0] < rrThreshold) {
            Float q = std::max((Float).05, 1 - Tr[0]);
            if (sampler.Get1D() < q) return 0;
            Tr /= 1 - q;
        }
    }
    return Spectrum(Tr);
}



// sample a heterogeneous point 
Spectrum HeterogeneousMedium::Sample(const Ray &ray, Sampler &sampler,
                                   MemoryArena &arena, MediumInteraction *mi, int cha
                                     ) const {
    ProfilePhase _(Prof::MediumSample);
    Spectrum omega = Spectrum(1.0f);
    return omega;
}

Spectrum HeterogeneousMedium::Sample(const Ray &ray, Sampler &sampler,
                                     MemoryArena &arena, MediumInteraction *mi,
                                     Spectrum w_front, int channel) const {
    ProfilePhase _(Prof::MediumSample);
    Spectrum omega = Spectrum(1.0f);
    Float tmax = INFINITY;
    Float tmin = 0.0;

    // normalize ray
    Ray rayNormalize = Ray(ray.o, Normalize(ray.d), ray.tMax * ray.d.Length());

    // find intersection? Robustness? Need not to intersect, the tmax was
    // compute before
    tmax = rayNormalize.tMax;
    // begin sample
    Float t = tmin;
    while (true) {
        // find a point by max miuT
        Float dist = -std::log(1 - sampler.Get1D()) / maxT;
        t += std::min(dist / ray.d.Length(), ray.tMax);
        if (t >= ray.tMax) break;
        // get  ��t ��n
        Spectrum miuT = sigma_t;
        Spectrum miuS = MiuS(rayNormalize(t));
        Spectrum miuN = Spectrum(maxT) - miuT;
        // decide which behavior by spectrum probility
        Float Pa_f = ((sigma_a[0] * w_front[0] + sigma_a[1] * w_front[1] +
                     sigma_a[2] * w_front[2]) * (1.0 / 3.0));
        Float Ps_f = ((sigma_s[0] * w_front[0] + sigma_s[1] * w_front[1] +
                     sigma_s[2] * w_front[2]) * (1.0 / 3.0));
        Float Pn_f = ((miuN[0] * w_front[0] + miuN[1] * w_front[1] +
                     miuN[2] * w_front[2]) * (1.0 / 3.0));
        Float c = Pa_f + Ps_f + Pn_f;
        Float Pa = Pa_f / c;
        Float Ps = Ps_f / c + Pa;
        Float Pn = Pn_f / c;

        Float si = sampler.Get1D();
        if (si < Ps) {
            // sample scattering
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = ARENA_ALLOC(arena, HenyeyGreenstein)(g);
            *mi = MediumInteraction(rayNormalize(t), -rayNormalize.d, ray.time,
                                    this, phase);
            // update omiga
            omega *= (miuS / (maxT * Ps));
            return omega;
        } else {
            // sample null-collision
            omega *= (miuN / (maxT * Pn));
        }
    }
    return omega;
}

Spectrum HeterogeneousMedium::SampleKd(
    const Ray &ray, Sampler &sampler, MemoryArena &arena, MediumInteraction *mi,
    Spectrum w_front, std::shared_ptr<Texture<Spectrum>> &Kd,
    SurfaceInteraction &isect, const Scene &scene, int flag,
    const std::vector<Vector3f> &colorvec,
    const std::vector<Vector3f> &sig_aouter,
    const std::vector<Vector3f> &sig_ainner) const {

	ProfilePhase _(Prof::MediumSample);
    Spectrum omega = Spectrum(1.0f);
    Float tmax = INFINITY;
    Float tmin = 0.0f;
    // find intersection? Robustness? Need not to intersect, the tmax was
    // compute before
    tmax = ray.tMax;
    // begin sample
    Float t = tmin;
	// Optimize. Only need one intersection per Tr and sample. Because ray.d doesn't change during compute
    Ray ray_reverse = Ray(ray(ray.tMax * 0.5), -ray.d);  // Ray origin is the middle of ray
    SurfaceInteraction isect_reverse;
    bool foundReverse = scene.Intersect(ray_reverse, &isect_reverse);
    // Get uv onece
    Point2f uv_reverse, uv_front;
    uv_front = isect.uv;
    uv_reverse = isect_reverse.uv;

    while (true) {
        // find a point by max miuT
        Float dist = -std::log(1 - sampler.Get1D()) / maxT;
        t += std::min(dist / ray.d.Length(), ray.tMax);
        if (t >= ray.tMax) break;

		// interpolate to get uv at ray(t)
        Point2f uv{0.0, 0.0};
        Float bilinear = (ray.tMax - t) / (ray.tMax * 0.5 + ray_reverse.tMax);
        // Float bilinear = 1.0;
        if (bilinear > 0.99) {
            uv = uv_reverse;
        } else if (bilinear <= 0.01) {
            uv = uv_front;
        } else {
            uv = bilinear * uv_reverse + (1.0 - bilinear) * uv_front;
        }
        // Get albedo value
        SurfaceInteraction isect_albedo;
        isect_albedo.uv = uv;
        Spectrum albedo = Kd->Evaluate(isect_albedo);
        // Albedo2sigma
        Spectrum sigma_a_albedo;
        FindCloest(albedo, sigma_a_albedo, flag, colorvec, sig_aouter, sig_ainner);
        sigma_a_albedo *= scale;
        // get  ��t ��n
        Spectrum miuT = sigma_a_albedo + sigma_s;
        Spectrum miuS = MiuS(ray(t));
        Spectrum miuN = Spectrum(maxT) - miuT;
        // decide which behavior by spectrum probility
        Float Pa_f = ((sigma_a_albedo[0] * w_front[0] + sigma_a_albedo[1] * w_front[1] +
					   sigma_a_albedo[2] * w_front[2]) * (1.0 / 3.0));
        Float Ps_f = ((sigma_s[0] * w_front[0] + sigma_s[1] * w_front[1] +
                       sigma_s[2] * w_front[2]) * (1.0 / 3.0));
        Float Pn_f = ((miuN[0] * w_front[0] + miuN[1] * w_front[1] +
                       miuN[2] * w_front[2]) * (1.0 / 3.0));
        Float c = Pa_f + Ps_f + Pn_f;
        Float Pa = Pa_f / c;
        Float Ps = Ps_f / c + Pa;
        Float Pn = Pn_f / c;

        Float si = sampler.Get1D();
        if (si < Ps) {
            // sample scattering
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = ARENA_ALLOC(arena, HenyeyGreenstein)(g);
            *mi = MediumInteraction(ray(t), -ray.d, ray.time,
                                    this, phase);
            // update omiga
            omega *= (miuS / (maxT * Ps));
            return omega;
        } else {
            // sample null-collision
            omega *= (miuN / (maxT * Pn));
        }
    }
    return omega;

}


}  // namespace pbrt
