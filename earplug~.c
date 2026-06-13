/* RT binaural filter: earplug~        */
/* based on KEMAR impulse measurement  */
/* Pei Xiang, summer 2004              */
/* Revised in fall 2006 by Jorge Castellanos */
/* Revised in spring 2009 by Hans-Christoph Steiner to compile in the data file */
/* Updated in 2020-2021 by Dan Wilcox & Chikashi Miyama */

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <errno.h>

/* impulse response data */
#ifdef EARPLUG_DATA_NO_EMBED
t_float earplug_impulses[368][2][128] = {{{0.0f}}};
#else
#include "earplug_data.h"
#endif

#define VERSION "0.3.0"

/* these pragmas only apply to Microsoft's compiler */
#ifdef _MSC_VER
#pragma warning( disable : 4244 ) /* uncast float/int conversion etc. */
#pragma warning( disable : 4305 ) /* uncast const double to float */
#endif

/* elevation degree:       -40  -30  -20  -10   0   10  20  30  40  50  60  70  80  90 */
/* index array:              0    1    2    3   4    5   6   7   8   9  10  11  12  13 */
/* impulse response number: 29   31   37   37  37   37  37  31  29  23  19  13   7   1 */ 
/* 0 degree response index:  0   29   60   97 134  171 208 245 276 305 328 347 360 367 */

static t_class *earplug_class;

/* number of azimuth samples per elevation ring from -40..80 degrees */
static const unsigned earplug_azim_count[13] = {29, 31, 37, 37, 37, 37, 37, 31, 29, 23, 19, 13, 7};

typedef struct _earplug
{
    t_object x_obj; 
    t_outlet *left_channel;
    t_outlet *right_channel;

    t_float azi;
    t_float ele;
    unsigned ch_L;
    unsigned ch_R;

    unsigned int azimOffset[13];

    t_float ir[2][128];
    t_float convBuffer[128];
    t_float (*impulses)[2][128];     /* a 3D array of 368x2x128 */
    t_float f;                       /* dummy float for dsp */
    int bufferPin;
} t_earplug;

static t_int *earplug_perform(t_int *w)
{
    t_earplug *x = (t_earplug *)(w[1]);
    t_float *in = (t_float *)(w[2]);
    t_float *right_out = (t_float *)(w[3]);
    t_float *left_out = (t_float *)(w[4]);
    int blocksize = (int)(w[5]);
    unsigned i;


    if (x->ele < 8.0) /* if elevation is less than 80 degrees... */
    { 
        /* a quantized version of the elevation */
        int elevInt = (int)floor(x->ele);
          /* index into elevation rings (-40..80 by 10 deg)
              adding 4 because the lowest elevation is -4 (in 10-degree units) */
        unsigned elevGridIndex = elevInt + 4;
        float elevFracUp = x->ele - (float)elevInt;
        float elevFracDown = 1.0f - elevFracUp;

        unsigned downCount = earplug_azim_count[elevGridIndex];
        unsigned upCount = earplug_azim_count[elevGridIndex + 1];
        unsigned lowerBase = x->azimOffset[elevGridIndex];
        unsigned upperBase = x->azimOffset[elevGridIndex + 1];

        float downPos = x->azi * (float)(downCount - 1) / 180.0f;
        unsigned downIdx0 = (unsigned)floorf(downPos);
        unsigned downIdx1 = downIdx0 + 1;
        if (downIdx1 >= downCount)
            downIdx1 = downIdx0;
        float downFrac = (downIdx1 == downIdx0) ? 0.0f : (downPos - (float)downIdx0);
        float downW0 = 1.0f - downFrac;
        float downW1 = downFrac;

        float upPos = x->azi * (float)(upCount - 1) / 180.0f;
        unsigned upIdx0 = (unsigned)floorf(upPos);
        unsigned upIdx1 = upIdx0 + 1;
        if (upIdx1 >= upCount)
            upIdx1 = upIdx0;
        float upFrac = (upIdx1 == upIdx0) ? 0.0f : (upPos - (float)upIdx0);
        float upW0 = 1.0f - upFrac;
        float upW1 = upFrac;
        
        for (i = 0; i < 128; i++)
        {
            /* bilinear interpolation in azimuth/elevation with clamped azimuth neighbors */
            t_float lowerL = downW0 * x->impulses[lowerBase + downIdx0][0][i]
                           + downW1 * x->impulses[lowerBase + downIdx1][0][i];
            t_float lowerR = downW0 * x->impulses[lowerBase + downIdx0][1][i]
                           + downW1 * x->impulses[lowerBase + downIdx1][1][i];
            t_float upperL = upW0 * x->impulses[upperBase + upIdx0][0][i]
                           + upW1 * x->impulses[upperBase + upIdx1][0][i];
            t_float upperR = upW0 * x->impulses[upperBase + upIdx0][1][i]
                           + upW1 * x->impulses[upperBase + upIdx1][1][i];

            x->ir[x->ch_L][i] = elevFracDown * lowerL + elevFracUp * upperL;
            x->ir[x->ch_R][i] = elevFracDown * lowerR + elevFracUp * upperR;
        }
    }
    else
    {
        /* if elevation is 80 degrees or more the interpolation requires only
           three points (because there's only one HRIR at 90 deg) */

        /* interpolate around the 80-degree ring, then blend toward the single 90-degree HRIR */
        const unsigned elev80Base = x->azimOffset[12];
        const unsigned elev80Count = earplug_azim_count[12];
        float azimPos = x->azi * (float)(elev80Count - 1) / 180.0f;
        unsigned azimIdx0 = (unsigned)floorf(azimPos);
        unsigned azimIdx1 = azimIdx0 + 1;
        if (azimIdx1 >= elev80Count)
            azimIdx1 = azimIdx0;
        float azimFrac = (azimIdx1 == azimIdx0) ? 0.0f : (azimPos - (float)azimIdx0);
        float azimW0 = 1.0f - azimFrac;
        float azimW1 = azimFrac;

        float elevFracUp = x->ele - 8.0f;
        float elevFracDown = 1.0f - elevFracUp;
        for (i = 0; i < 128; i++)
        {
            /* elevFracDown: these two lines interpolate the lower two HRIRs
                 elevFracUp: multiply the 90 degree HRIR with its corresponding fraction */
            x->ir[x->ch_L][i] = elevFracDown *
                                        (azimW0 * x->impulses[elev80Base + azimIdx0][0][i] +
                                        azimW1 * x->impulses[elev80Base + azimIdx1][0][i])
                                        + elevFracUp * x->impulses[367][0][i];
            x->ir[x->ch_R][i] = elevFracDown *
                                        (azimW0 * x->impulses[elev80Base + azimIdx0][1][i] +
                                        azimW1 * x->impulses[elev80Base + azimIdx1][1][i])
                                        + elevFracUp * x->impulses[367][1][i]; 
        }
    }

    float inSample;
    float convSum[2]; /* to accumulate the sum during convolution */

    /* convolve the interpolated HRIRs (left and right) with the input signal */
    while (blocksize--)
    {
        convSum[0] = 0; 
        convSum[1] = 0; 

        inSample = *(in++);

        x->convBuffer[x->bufferPin] = inSample;
        for (i = 0; i < 128; i++)
        { 
            convSum[0] += x->ir[0][i] * x->convBuffer[(x->bufferPin - i) &127];
            convSum[1] += x->ir[1][i] * x->convBuffer[(x->bufferPin - i) &127];
        }   

        x->bufferPin = (x->bufferPin + 1) & 127;

        *left_out++ = convSum[0];
        *right_out++ = convSum[1];
    }
    return w + 6;
}

static void earplug_azimuth(t_earplug *x, float value) {
    if (value < 0 || value > 360)
        value = 0;
    if (value <= 180){
        x->ch_L = 0;
        x->ch_R = 1;
    }
    else{ 
        x->ch_L = 1;
        x->ch_R = 0;
        value = 360.0 - value;
    }
    x->azi = value;
}

static void earplug_elevation(t_earplug *x, float value) {

    if (value < -40)
        value = -40;
    if (value > 90)
        value = 90;
    /* divided by 10 since each elevation is 10 degrees apart */
    x->ele = value * 0.1;
}

static void earplug_dsp(t_earplug *x, t_signal **sp)
{
    /* callback, params, userdata, in_samples,   out_L,        out_R,        blocksize */
    dsp_add(earplug_perform, 5, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_n);
}

static void *earplug_new(t_floatarg azimArg, t_floatarg elevArg)
{
    t_earplug *x = (t_earplug *)pd_new(earplug_class);
    x->left_channel = outlet_new(&x->x_obj, gensym("signal"));
    x->right_channel = outlet_new(&x->x_obj, gensym("signal"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("azimuth"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("elevation"));

    x->ch_L = 0;
    x->ch_R = 1;
    earplug_azimuth(x, azimArg);
    earplug_elevation(x, elevArg);

    int i, j;
    FILE *fp;
    t_symbol *canvasdir = canvas_getdir(canvas_getcurrent());
    char buff[MAXPDSTRING], *bufptr;
    int filedesc;

    filedesc = open_via_path(canvasdir->s_name, "earplug_data.txt", "", buff, &bufptr, MAXPDSTRING, 0);
    if (filedesc >= 0) /* if there was no error opening the text file... */
    {
        int ret;
        fp = fdopen(filedesc, "r");
        for (i = 0; i < 368; i++) 
        {
            do {ret = fgetc(fp);}
            while (ret != 10 && ret != EOF);
            if (ret != EOF)
            {
                for (j = 0; j < 128; j++)
                {
                    ret = fscanf(fp, "%f %f ", &earplug_impulses[i][0][j],
                                               &earplug_impulses[i][1][j]);
                    if (ret == EOF) {break;}
                }
            }
            if (ret == EOF)
            {
                pd_error(x, "earplug~: could not load %s/earplug_data.txt, check format?", buff);
                break;
            }
        }
        fclose(fp);
        if (ret != EOF) {logpost(x, 3, "earplug~: loaded %s/earplug_data.txt", buff);}
    }
    x->impulses = earplug_impulses;

    for (i = 0; i < 128; i++)
         x->convBuffer[i] = 0.f; 
    x->bufferPin = 0;

    x->azimOffset[0] = 0; 
    x->azimOffset[1] = 29;
    x->azimOffset[2] = 60;
    x->azimOffset[3] = 97;
    x->azimOffset[4] = 134;
    x->azimOffset[5] = 171;
    x->azimOffset[6] = 208;
    x->azimOffset[7] = 245;
    x->azimOffset[8] = 276;
    x->azimOffset[9] = 305;
    x->azimOffset[10] = 328;
    x->azimOffset[11] = 347;
    x->azimOffset[12] = 360;

    return x;
}

void earplug_tilde_setup(void)
{
    earplug_class = class_new(gensym("earplug~"), (t_newmethod)earplug_new, 0,
        sizeof(t_earplug), CLASS_DEFAULT, A_DEFFLOAT, A_DEFFLOAT, 0);

    CLASS_MAINSIGNALIN(earplug_class, t_earplug, f);

    class_addmethod(earplug_class, (t_method)earplug_dsp, gensym("dsp"), A_CANT, 0);
    class_addmethod(earplug_class, (t_method)earplug_azimuth, gensym("azimuth"), A_FLOAT, 0);
    class_addmethod(earplug_class, (t_method)earplug_elevation, gensym("elevation"), A_FLOAT, 0);

    post("earplug~ %s: binaural filter with measured responses", VERSION);
    post("    elevation: -40 to 90 degrees, azimuth: 360 degrees");
    post("    do not use a blocksize > 8192");
}
