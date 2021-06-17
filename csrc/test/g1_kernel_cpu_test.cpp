#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <string>
#include <fftw3.h> // for all fttw related items

#include "g1_kernel.hpp"

class G1KernelCpuTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( G1KernelCpuTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_direct_unbiased_normalisation);
    CPPUNIT_TEST( test_direct_unbiased_normalisation_2_threads );
    CPPUNIT_TEST( test_direct_biased_normalisation );
    // CPPUNIT_TEST( test_fftw );
    CPPUNIT_TEST( test_fftw );

    CPPUNIT_TEST_SUITE_END();
private:
    const static int N = 101;
    const int tau_points = 50;

    short* chA_data;
    short* chB_data;
    double** data_out;

    short test_data[N] = {225, 196, 268, 225, 269, 333, 201, 293, 303, 186,
                          213, 172, 206, 206, 338, 455, 251, 300, 297, 199,
                          191, 162, 226, 262, 288, 266, 358, 292, 296, 295,
                          221, 251, 242, 298, 258, 276, 272, 251, 280, 185,
                          200, 206, 229, 259, 185, 308, 311, 333, 327, 304,
                          58, 192, 191, 243, 282, 194, 351, 242, 317, 298,
                          209, 308, 103, 227, 215, 262, 308, 353,305, 270,
                          282, 161, 261, 187, 280, 319, 270, 429, 311, 329,
                          220, 211, 174, 175, 129, 288, 355, 256, 339, 345,
                          389, 281, 109, 161, 243, 239, 257, 290, 205, 272,
                          202
    };

    double expected_g1_biased_normalisation[N] = {1.0, 0.27600544998591797, 0.15918176205175, -0.11336260837505102, -0.4021687358135168, -0.3249860282662008, -0.46586828862555396, -0.1502429202912954, 0.0012467754402799072, 0.25451548730153556, 0.310908207503377, 0.3975154147225708, 0.1859997254389784, 0.01542799156474954, -0.1477309192262582, -0.39848224527651316, -0.17310991538336223, -0.3179520938676653, 0.07360318538651384, 0.16896645459147713, 0.23982236673212293, 0.30173438149168186, 0.0950788051238177, 0.030672807038568436, -0.21248992009062984, -0.23391060453638163, -0.34091055914572466, -0.19856632450598968, -0.06220692339296371, 0.10181165130751317, 0.30334824358762097, 0.22991268521297248, 0.24208563364127175, 0.04436739087320818, -0.016098803182965186, -0.28586041959576924, -0.30577045534101394, -0.19499900994297892, -0.09086777494550671, 0.024253766437260883, 0.08014830676452796, 0.30284917767416986, 0.19428688736795358, 0.19491167475194324, 0.0486981777312503, -0.16617560880871265, -0.1397543377157834, -0.247265065759728, -0.0622802383736077, 0.01817063150140414, 0.07271741615459379, 0.16847477923800555, 0.15777459639152464, 0.11580144448243662, 0.012105408121219902, -0.10942373224148706, -0.1415076465135238, -0.13297450393435914, -0.13198343530448994, 0.027129413762845753, 0.1004559109295351, 0.13492622463822598, 0.22948938002944672, 0.12440858606778292, 0.03407760991112568, -0.0759218369952974, -0.19422818299655503, -0.15460267546563278, -0.12889541027601267, -0.11895552699161839, 0.03748481733722343, 0.12876223281696492, 0.13490175348693076, 0.11883833832660298, 0.03715951616043774, 0.02272135999690698, -0.059302539064067294, -0.14287346574894874, -0.10928603624852544, -0.05308604646533355, 0.0005338655199175186, 0.07208662873210395, 0.06922340049046714, 0.04829766965391171, 0.008983405418411828, -0.036909636674832706, -0.003588207192298495, -0.023386302653979923, -0.03369777614767774, 0.004256058665216487, -0.01323114078157908, 0.017913672076111994, 0.031790466875795, -0.006412782814054045, 0.008841226800123378, -0.0033390776013569553, -0.008691291706744897, 0.0087345062592617, -2.698034642361048e-05, 0.006192351453962012, 0.0037789536064417673};


    double expected_g1_unbiased_normalisation[N] = {1.0, 0.27876550448577714, 0.16239755522451263, -0.11683289230489952, -0.4187530135790226, -0.34191238390506545, -0.4952915489597995, -0.16143122286617909, 0.001354024940519039, 0.2794137414940771, 0.34507394459166024, 0.4461006320775517, 0.21107834010490809, 0.01770712668226936, -0.17150371082588595, -0.467984962475905, -0.2056953112202304, -0.3822995414361214, 0.08956532197636022, 0.208117218460234, 0.2990377659252397, 0.3809396566332483, 0.12155644705703274, 0.03971735270378734, -0.2787205445344625, -0.3108548823444019, -0.45909288631624257, -0.27101619966358054, -0.08606711318752513, 0.1428191219730393, 0.43152355777957346, 0.33173116009300313, 0.35435723185171664, 0.06589862467932392, -0.024268345096708714, -0.43745306635110137, -0.47512024599142166, -0.3077328125662636, -0.14567690903962188, 0.03951016790586047, 0.13270457349536596, 0.5097961157515193, 0.33259280718920864, 0.33941515775769426, 0.0862897535237944, -0.29970958017285676, -0.25663978380534774, -0.4624772526246765, -0.118684982561026, 0.03529295733926573, 0.14400900061988184, 0.34031905406077123, 0.3252088619498773, 0.24366553943179373, 0.02601374936687681, -0.24025645557369985, -0.3176060510636867, -0.30523692948568804, -0.3100076038547322, 0.06523978071541478, 0.24746456107031817, 0.3406887172115206, 0.5943186508454902, 0.3306649261275283, 0.09302266489253225, -0.21300293157013994, -0.5604870423614874, -0.45926088888320327, -0.39449807387506913, -0.37545338206729556, 0.12212795325998603, 0.4334995171504485, 0.4698302449027589, 0.4286668632495322, 0.1390041160075634, 0.08826374460336942, -0.23958225781883186, -0.6012591683601592, -0.4799082461348291, -0.24371321331812218, 0.0025676389291271135, 0.3640374750971249, 0.36797702365985163, 0.2710035908358379, 0.05337199689762322, -0.23299208150988146, -0.02416059509480987, -0.16871546914656946, -0.2618057993011886, 0.03582182709890543, -0.12148592899449882, 0.18092808796873114, 0.35675968382836615, -0.08096138302743232, 0.12756627240178017, -0.05620780628950875, -0.1755640924762469, 0.22054628304635793, -0.0009083383295948861, 0.3127137484250816, 0.3816743142506185};

public:
    void setUp() {
        chA_data = new short[N];
        chB_data = new short[N];

        for (int i(0); i < N; i++) {
            chA_data[i] = test_data[i];
            chB_data[i] = test_data[i];
        }

        data_out = new double*[G1::no_outputs];
        for (int i(0); i < G1::no_outputs; i++)
            data_out[i] = new double[tau_points]();
    }
    void tearDown() {
        delete[] chA_data;
        delete[] chB_data;

        for (int i(0); i < G1::no_outputs; i++)
            delete[] data_out[i];
        delete[] data_out;
    }
    void beforeEach() {
        for (int i(0); i < G1::no_outputs; i++) {
            for (int t; t < tau_points; t)
                data_out[i][t] = 0;
        }
    }

    void test_direct_unbiased_normalisation() {
        int no_threads = 1;

        G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, true, no_threads);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 expected_g1_unbiased_normalisation[tau], data_out[CHAG1][tau], 0.001);
        }
    }

    void test_direct_unbiased_normalisation_2_threads() {
        int no_threads = 2;

        G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, true, no_threads);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 expected_g1_unbiased_normalisation[tau], data_out[CHAG1][tau], 0.001);
        }
    }

    void test_direct_biased_normalisation() {
        int no_threads = 1;

        G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 expected_g1_biased_normalisation[tau], data_out[CHAG1][tau], 0.001);
        }
    }

    void test_fftw(){

        // Generate plans
        int time_limit = 1;
        int no_threads = 8;
        G1::CPU::FFTW::g1_prepare_fftw_plan("./dump/test-fftw-plan", time_limit, no_threads);

        // Pre-kernel setup
        double **data_out = new double*[G1::no_outputs]; fftw_complex *aux_array;
        fftw_plan *plans_forward = new fftw_plan[G1::no_outputs]; fftw_plan *plans_backward = new fftw_plan[G1::no_outputs];

        G1::CPU::FFTW::g1_allocate_memory(data_out, aux_array, "./dump/test-fftw-plan", plans_forward, plans_backward);

        G1::CPU::FFTW::g1_kernel(chA_data, chB_data,
                                 data_out, aux_array,
                                 plans_forward, plans_backward);

        for (int tau(0); tau < 20; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 expected_g1_biased_normalisation[tau],
                                                 data_out[CHAG1][tau], 0.05);
        }

        // Post kernel
        G1::CPU::FFTW::g1_free_memory(data_out, plans_forward, plans_backward);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelCpuTest );
